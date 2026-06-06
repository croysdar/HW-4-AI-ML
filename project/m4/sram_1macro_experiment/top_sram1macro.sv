// =============================================================================
// File   : project/m4/sram_1macro_experiment/top_sram1macro.sv
// Module : bnn_top  (narrow-compute, single-SRAM-macro variant)
//
// One 32-bit-wide SRAM macro feeds a 32-bit-wide compute_core. Each 256-bit
// AXI activation beat is serialized into 8 × 32-bit chunks; each chunk drives
// one SRAM read + one compute cycle. The signed dot product accumulates
// across all chunks of all beats in a filter tile, then is emitted on the AXI
// master interface — identical observable behaviour to the 256-bit baseline,
// only 8× slower per tile.
//
// Memory
// ------
// 1 × sky130_sram_1kbyte_1rw1r_32x256_8 = 32 bits × 256 entries = 1 KB.
// At 8 reads per beat that holds 32 beats of weights → ~6 conv4 filters
// resident at a time (5 beats/filter). Host reloads via the sideband w_*
// port between tile groups (~21 reloads/frame at 128 filters, negligible).
//
// Pipeline alignment
// ------------------
// SRAM read latency = 1 cycle. The activation chunk is muxed combinationally
// from a registered activation buffer, then both activation chunk and weight
// chunk are registered once before entering compute_core to share the same
// pipeline depth as the SRAM data path.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH    = 256,        // AXI activation width (host-facing)
    parameter int CHUNK_WIDTH     = 32,         // compute / SRAM width
    parameter int CHUNKS_PER_BEAT = 8,          // VECTOR_WIDTH / CHUNK_WIDTH
    parameter int WEIGHT_DEPTH    = 256,        // SRAM entries (1 macro)
    parameter int MAX_BEATS       = 5
) (
    input  logic        clk,
    input  logic        rst,

    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic [VECTOR_WIDTH-1:0] s_axis_tdata,
    input  logic        s_axis_tlast,

    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tlast,

    // Sideband weight load — host writes one 32-bit chunk per w_en cycle.
    input  logic        w_en,
    input  logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr,
    input  logic [CHUNK_WIDTH-1:0] w_data,

    input  logic [3:0]  cfg_beats_per_tile,
    output logic [15:0] tile_count
);

    // =========================================================================
    // axis_interface (unchanged — still 256-bit AXI on the host side)
    // =========================================================================
    logic        core_valid, core_ready;
    logic [VECTOR_WIDTH-1:0] core_data;
    logic        frame_done;

    axis_interface u_axis_if (
        .aclk          (clk),
        .aresetn       (~rst),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tlast  (s_axis_tlast),
        .core_valid    (core_valid),
        .core_ready    (core_ready),
        .core_data     (core_data),
        .frame_done    (frame_done)
    );

    // =========================================================================
    // Activation buffer + chunk counter
    // -------------------------------------------------------------------------
    // On a new beat: latch core_data and start streaming chunks 0..7.
    // beat_consumed pulses high in the cycle we accept the final chunk;
    // that's the cycle we backpressure-release the axis_interface.
    // =========================================================================
    localparam int CHUNK_CTR_W = $clog2(CHUNKS_PER_BEAT);  // 3

    logic [VECTOR_WIDTH-1:0] act_buf;
    logic                    act_buf_valid;
    logic [CHUNK_CTR_W-1:0]  chunk_ctr;

    logic                    beat_start;   // first chunk of a new beat
    logic                    beat_consumed; // last chunk of current beat
    logic                    chunk_valid;   // a chunk is being driven into compute this cycle

    assign chunk_valid   = act_buf_valid;
    assign beat_consumed = chunk_valid && (chunk_ctr == CHUNK_CTR_W'(CHUNKS_PER_BEAT - 1));
    assign beat_start    = core_valid && core_ready;

    // axis_interface's core_ready is wired to "ready for next beat" — we only
    // accept a new 256-bit beat when the current buffer is empty OR we're on
    // the last chunk this cycle.
    assign core_ready = (!act_buf_valid) || beat_consumed;

    always_ff @(posedge clk) begin
        if (rst) begin
            act_buf       <= '0;
            act_buf_valid <= 1'b0;
            chunk_ctr     <= '0;
        end else begin
            if (beat_start) begin
                act_buf       <= core_data;
                act_buf_valid <= 1'b1;
                chunk_ctr     <= '0;
            end else if (chunk_valid) begin
                if (beat_consumed) begin
                    act_buf_valid <= 1'b0;
                    chunk_ctr     <= '0;
                end else begin
                    chunk_ctr <= chunk_ctr + 1'b1;
                end
            end
        end
    end

    // Combinational chunk extraction from the buffer.
    logic [CHUNK_WIDTH-1:0] act_chunk;
    assign act_chunk = act_buf[chunk_ctr*CHUNK_WIDTH +: CHUNK_WIDTH];

    // =========================================================================
    // Weight address counter — increments every chunk_valid cycle
    // =========================================================================
    localparam int W_ADDR_W = $clog2(WEIGHT_DEPTH);  // 8 for 256 entries
    logic [W_ADDR_W-1:0] w_ptr;
    logic [W_ADDR_W-1:0] sram_addr;

    assign sram_addr = w_en ? w_addr : w_ptr;

    always_ff @(posedge clk) begin
        if (rst)
            w_ptr <= '0;
        else if (chunk_valid) begin
            if (w_ptr == W_ADDR_W'(WEIGHT_DEPTH - 1))
                w_ptr <= '0;
            else
                w_ptr <= w_ptr + 1'b1;
        end
    end

    // =========================================================================
    // SRAM macro — single instance
    // =========================================================================
    logic [CHUNK_WIDTH-1:0] weight_chunk;

    sky130_sram_1kbyte_1rw1r_32x256_8 u_bank (
        .clk0   (clk),
        .csb0   (1'b0),
        .web0   (~w_en),
        .wmask0 (4'b1111),
        .addr0  (sram_addr),
        .din0   (w_data),
        .dout0  (weight_chunk),
        .clk1   (clk),
        .csb1   (1'b1),
        .addr1  ('0),
        .dout1  ()
    );

    // =========================================================================
    // Pipeline alignment register — delay act_chunk by 1 cycle to match the
    // 1-cycle SRAM read latency on weight_chunk.
    // =========================================================================
    logic [CHUNK_WIDTH-1:0] act_chunk_r;
    logic                   chunk_valid_r;
    always_ff @(posedge clk) begin
        if (rst) begin
            act_chunk_r   <= '0;
            chunk_valid_r <= 1'b0;
        end else begin
            act_chunk_r   <= act_chunk;
            chunk_valid_r <= chunk_valid;
        end
    end

    // =========================================================================
    // Tile beat counter — counts BEATS (not chunks). Increments on beat_consumed.
    // =========================================================================
    localparam int BEAT_CTR_W = $clog2(MAX_BEATS + 1);
    logic [BEAT_CTR_W-1:0] beat_ctr;
    logic [3:0]            beats_this_tile;
    logic                  last_beat;     // marks the LAST CHUNK of the last beat

    assign last_beat = beat_consumed
                     && (beat_ctr == BEAT_CTR_W'(beats_this_tile - 1));

    always_ff @(posedge clk) begin
        if (rst) begin
            beat_ctr        <= '0;
            beats_this_tile <= 4'd5;
        end else if (beat_consumed) begin
            if (last_beat) begin
                beat_ctr        <= '0;
                beats_this_tile <= cfg_beats_per_tile;
            end else begin
                beat_ctr <= beat_ctr + 1'b1;
            end
        end else if (beat_ctr == '0 && !chunk_valid) begin
            beats_this_tile <= cfg_beats_per_tile;
        end
    end

    // =========================================================================
    // Pipeline delay — last_beat occurs at the chunk_valid cycle of the final
    // chunk; compute_core latency = 3 chunks; alignment register adds 1. So the
    // accumulator finishes updating 4 cycles after last_beat is asserted.
    // =========================================================================
    logic last_beat_r1, last_beat_r2, last_beat_r3, last_beat_r4;
    always_ff @(posedge clk) begin
        if (rst) begin
            last_beat_r1 <= 1'b0;
            last_beat_r2 <= 1'b0;
            last_beat_r3 <= 1'b0;
            last_beat_r4 <= 1'b0;
        end else begin
            last_beat_r1 <= last_beat;
            last_beat_r2 <= last_beat_r1;
            last_beat_r3 <= last_beat_r2;
            last_beat_r4 <= last_beat_r3;
        end
    end

    // =========================================================================
    // compute_core (32-bit narrow variant)
    // =========================================================================
    logic signed [31:0] accum_out;
    logic               accum_clear;

    // accum_clear fires after the tile result is captured, clearing accum
    // for the next tile. last_beat_r4 = "result ready"; clear is one cycle later.
    always_ff @(posedge clk)
        accum_clear <= rst ? 1'b0 : last_beat_r4;

    compute_core #(.VECTOR_WIDTH(CHUNK_WIDTH)) u_core (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (chunk_valid_r),
        .s_ready    (/* unused — always 1 */),
        .accum_clear(accum_clear),
        .act_in     (act_chunk_r),
        .weight_in  (weight_chunk),
        .accum_out  (accum_out)
    );

    // =========================================================================
    // AXI4-Stream master output — emit tile result at last_beat_r4
    // =========================================================================
    logic [31:0] result_reg;
    logic        result_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            result_reg   <= '0;
            result_valid <= 1'b0;
        end else begin
            if (last_beat_r4 && (!result_valid || m_axis_tready)) begin
                result_reg   <= accum_out;
                result_valid <= 1'b1;
            end else if (result_valid && m_axis_tready) begin
                result_valid <= 1'b0;
            end
        end
    end

    assign m_axis_tvalid = result_valid;
    assign m_axis_tdata  = result_reg;
    assign m_axis_tlast  = result_valid;

    // =========================================================================
    // Tile counter
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst)
            tile_count <= '0;
        else if (last_beat_r4)
            tile_count <= tile_count + 1'b1;
    end

endmodule

`default_nettype wire
