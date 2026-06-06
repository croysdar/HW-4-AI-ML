// =============================================================================
// File   : project/m4/sram_8macro_experiment/top_sram8macro.sv
// Module : bnn_top  (256-bit wide, 8-SRAM-macro, 40 MHz variant)
//
// 8 × sky130_sram_1kbyte_1rw1r_32x256_8 in parallel → 256-bit weight reads.
// AXI beats feed the 256-bit compute_core directly (no chunk serialisation).
// Each beat = 1 compute cycle.  Tiles pipeline back-to-back: throughput is
// one tile result per n_beats cycles (sustained).
//
// Memory
// ------
// 8 × 256-entry × 32-bit = 256 logical 256-bit weight words.
// conv4 capacity: 256/5 = 51 filters resident (vs 6 in 1-macro design).
//
// Write interface
// ---------------
// 8 consecutive w_en cycles at the same w_addr load one 256-bit word.
// An internal 3-bit bank counter (w_bank_sel) routes each 32-bit w_data
// to banks 0..7 sequentially.  Host keeps w_addr stable for all 8 cycles.
//
// Pipeline
// --------
// SRAM read latency = 1 cycle.  act_in is registered once before
// compute_core to match.  compute_core has 3 internal stages.
// last_beat delayed 4 cycles (1 SRAM + 3 compute) before result capture.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH = 256,
    parameter int WEIGHT_DEPTH = 256,   // logical 256-bit words (= SRAM rows)
    parameter int MAX_BEATS    = 5
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

    // Sideband weight load — 8 consecutive w_en cycles per logical word.
    // w_addr must be stable for all 8 cycles.
    input  logic        w_en,
    input  logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr,
    input  logic [31:0] w_data,

    input  logic [3:0]  cfg_beats_per_tile,
    output logic [15:0] tile_count
);

    // =========================================================================
    // AXI4-Stream interface
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

    assign core_ready = 1'b1;  // always accept — no chunk buffering needed

    // =========================================================================
    // Beat valid
    // =========================================================================
    logic beat_valid;
    assign beat_valid = core_valid & core_ready;

    // =========================================================================
    // Weight read address — increments every beat_valid, wraps at WEIGHT_DEPTH
    // =========================================================================
    localparam int W_ADDR_W = $clog2(WEIGHT_DEPTH);  // 8
    logic [W_ADDR_W-1:0] r_ptr;

    always_ff @(posedge clk) begin
        if (rst)
            r_ptr <= '0;
        else if (beat_valid) begin
            if (r_ptr == W_ADDR_W'(WEIGHT_DEPTH - 1))
                r_ptr <= '0;
            else
                r_ptr <= r_ptr + 1'b1;
        end
    end

    // =========================================================================
    // Write bank counter — auto-increments per w_en cycle, routes to bank 0..7
    // =========================================================================
    logic [2:0] w_bank_sel;
    always_ff @(posedge clk) begin
        if (rst)
            w_bank_sel <= '0;
        else if (w_en)
            w_bank_sel <= w_bank_sel + 1'b1;
    end

    // =========================================================================
    // 8 × sky130_sram_1kbyte_1rw1r_32x256_8 — parallel 256-bit weight read
    // =========================================================================
    logic [VECTOR_WIDTH-1:0] weight_word;  // {bank7, ..., bank0}

    generate
        for (genvar i = 0; i < 8; i++) begin : gen_banks
            logic bank_wen;
            assign bank_wen = w_en && (w_bank_sel == 3'(i));

            sky130_sram_1kbyte_1rw1r_32x256_8 u_bank (
                .clk0   (clk),
                .csb0   (1'b0),
                .web0   (~bank_wen),
                .wmask0 (4'b1111),
                .addr0  (bank_wen ? w_addr : r_ptr),
                .din0   (w_data),
                .dout0  (weight_word[i*32 +: 32]),
                .clk1   (clk),
                .csb1   (1'b1),
                .addr1  ('0),
                .dout1  ()
            );
        end
    endgenerate

    // =========================================================================
    // Pipeline alignment — register act by 1 cycle to match SRAM read latency
    // =========================================================================
    logic [VECTOR_WIDTH-1:0] act_r;
    logic                    beat_valid_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            act_r        <= '0;
            beat_valid_r <= 1'b0;
        end else begin
            act_r        <= core_data;
            beat_valid_r <= beat_valid;
        end
    end

    // =========================================================================
    // Tile beat counter — counts beats per tile, fires last_beat on final beat
    // =========================================================================
    localparam int BEAT_CTR_W = $clog2(MAX_BEATS + 1);

    logic [BEAT_CTR_W-1:0] beat_ctr;
    logic [3:0]            beats_this_tile;
    logic                  last_beat;

    assign last_beat = beat_valid
                     && (beat_ctr == BEAT_CTR_W'(beats_this_tile - 1));

    always_ff @(posedge clk) begin
        if (rst) begin
            beat_ctr        <= '0;
            beats_this_tile <= 4'd5;
        end else if (beat_valid) begin
            if (last_beat) begin
                beat_ctr        <= '0;
                beats_this_tile <= cfg_beats_per_tile;
            end else begin
                beat_ctr <= beat_ctr + 1'b1;
            end
        end else if (beat_ctr == '0 && !beat_valid) begin
            beats_this_tile <= cfg_beats_per_tile;
        end
    end

    // =========================================================================
    // Pipeline delay — last_beat at beat_valid cycle; result ready 4 cycles
    // later (1 SRAM + 3 compute stages).
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
    // compute_core (256-bit, original rtl/compute_core.sv)
    // =========================================================================
    logic signed [31:0] accum_out;
    logic               accum_clear;

    always_ff @(posedge clk)
        accum_clear <= rst ? 1'b0 : last_beat_r4;

    compute_core #(.VECTOR_WIDTH(VECTOR_WIDTH)) u_core (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (beat_valid_r),
        .s_ready    (),
        .accum_clear(accum_clear),
        .act_in     (act_r),
        .weight_in  (weight_word),
        .accum_out  (accum_out)
    );

    // =========================================================================
    // AXI4-Stream master output
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
