// =============================================================================
// File   : project/m4/rtl/top.sv
// Module : bnn_top  (256-bit wide, 4-SRAM-macro, 40 MHz — final M4 design)
//
// 4 × sky130_sram_1kbyte_1rw1r_32x256_8 with 2-phase reads → 256-bit weight.
// Lower 128 bits (phase 0): rows 0-127 across all 4 banks.
// Upper 128 bits (phase 1): rows 128-255 across all 4 banks.
// 128 logical 256-bit weight words total.
//
// Write interface
// ---------------
// 8 consecutive w_en cycles at the same w_addr load one 256-bit word.
// w_bank_sel[1:0] selects bank (0-3), w_bank_sel[2] selects half:
//   0 → rows 0-127 (lower 128-bit half),  1 → rows 128-255 (upper half).
//
// Pipeline
// --------
// Each incoming beat triggers a 2-cycle SRAM read sequence:
//   Phase 0 (beat cycle)    : issue read for addr {0, r_ptr} (lower half)
//   Phase 1 (stall cycle)   : issue read for addr {1, r_ptr} (upper half);
//                             latch lower-half result into weight_low_r
//   Two cycles later: upper-half result available; weight_word assembled.
//
// core_ready = ~read_phase  → AXI stalled for 1 cycle per beat.
// Activation pipeline: 2-cycle delay to match 2-phase SRAM read.
// last_beat delayed 5 cycles (2 SRAM phases + 3 compute stages).
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH = 256,
    parameter int WEIGHT_DEPTH = 128,   // logical 256-bit words
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
    // AXI4-Stream interface (skid buffer)
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
    // 2-phase SRAM read control
    // =========================================================================
    logic read_phase;   // 0 = phase 0 (lower half), 1 = phase 1 (upper half)

    // core_ready: accept a new beat only when not in the middle of phase 1
    assign core_ready = ~read_phase;

    logic beat_valid;
    assign beat_valid = core_valid & core_ready;

    always_ff @(posedge clk) begin
        if (rst)
            read_phase <= 1'b0;
        else if (read_phase)
            read_phase <= 1'b0;     // always return to phase 0 after phase 1
        else if (beat_valid)
            read_phase <= 1'b1;     // start phase 1 when beat accepted
    end

    // =========================================================================
    // Weight read address — increments at end of phase 1, wraps at WEIGHT_DEPTH
    // =========================================================================
    localparam int W_ADDR_W = $clog2(WEIGHT_DEPTH);  // 7 bits
    logic [W_ADDR_W-1:0] r_ptr;

    always_ff @(posedge clk) begin
        if (rst)
            r_ptr <= '0;
        else if (read_phase)   // increment when phase 1 completes
            r_ptr <= (r_ptr == W_ADDR_W'(WEIGHT_DEPTH - 1)) ? '0 : r_ptr + 1'b1;
    end

    // =========================================================================
    // Write bank counter — auto-increments per w_en cycle
    // w_bank_sel[1:0] → bank (0-3), w_bank_sel[2] → half (0=lower, 1=upper)
    // =========================================================================
    logic [2:0] w_bank_sel;
    always_ff @(posedge clk) begin
        if (rst)
            w_bank_sel <= '0;
        else if (w_en)
            w_bank_sel <= w_bank_sel + 1'b1;
    end

    // =========================================================================
    // 4 × sky130_sram_1kbyte_1rw1r_32x256_8
    // Each bank: 256 rows × 32 bits.
    // Rows   0-127: lower 128 bits of each logical 256-bit word.
    // Rows 128-255: upper 128 bits.
    // =========================================================================
    logic [127:0] sram_dout;    // combined 4-bank output

    generate
        for (genvar i = 0; i < 4; i++) begin : gen_banks
            logic        bank_wen;
            logic [7:0]  bank_addr;

            assign bank_wen  = w_en && (w_bank_sel[1:0] == 2'(i));
            assign bank_addr = bank_wen
                             ? {w_bank_sel[2], w_addr[W_ADDR_W-1:0]}  // write: {half, w_addr}
                             : {read_phase,    r_ptr[W_ADDR_W-1:0]};   // read: {phase, r_ptr}

            sky130_sram_1kbyte_1rw1r_32x256_8 u_bank (
                .clk0   (clk),
                .csb0   (1'b0),
                .web0   (~bank_wen),
                .wmask0 (4'b1111),
                .addr0  (bank_addr),
                .din0   (w_data),
                .dout0  (sram_dout[i*32 +: 32]),
                .clk1   (clk),
                .csb1   (1'b1),
                .addr1  ('0),
                .dout1  ()
            );
        end
    endgenerate

    // =========================================================================
    // Weight assembly — 2-phase latch
    // At posedge when read_phase=1: sram_dout = lower-half (phase-0 result)
    // Two cycles after beat: sram_dout = upper-half (phase-1 result)
    // =========================================================================
    logic [127:0] weight_low_r;   // latched lower 128 bits
    logic [255:0] weight_word;    // full 256-bit weight for compute_core

    always_ff @(posedge clk) begin
        if (read_phase)
            weight_low_r <= sram_dout;  // latch phase-0 result at end of phase 1
    end

    assign weight_word = {sram_dout, weight_low_r};  // assembled when beat_valid_r2

    // =========================================================================
    // Pipeline alignment — 2-cycle activation delay to match 2-phase SRAM read
    // =========================================================================
    logic [VECTOR_WIDTH-1:0] act_r1, act_r2;
    logic                    beat_valid_r1, beat_valid_r2;

    always_ff @(posedge clk) begin
        if (rst) begin
            act_r1        <= '0;
            act_r2        <= '0;
            beat_valid_r1 <= 1'b0;
            beat_valid_r2 <= 1'b0;
        end else begin
            act_r1        <= core_data;
            act_r2        <= act_r1;
            beat_valid_r1 <= beat_valid;
            beat_valid_r2 <= beat_valid_r1;
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
    // Pipeline delay — last_beat fires at beat_valid cycle.
    // 2-phase SRAM (2 cycles) + 3 compute stages = 5 delay registers.
    // =========================================================================
    logic last_beat_r1, last_beat_r2, last_beat_r3, last_beat_r4, last_beat_r5;

    always_ff @(posedge clk) begin
        if (rst) begin
            last_beat_r1 <= 1'b0;
            last_beat_r2 <= 1'b0;
            last_beat_r3 <= 1'b0;
            last_beat_r4 <= 1'b0;
            last_beat_r5 <= 1'b0;
        end else begin
            last_beat_r1 <= last_beat;
            last_beat_r2 <= last_beat_r1;
            last_beat_r3 <= last_beat_r2;
            last_beat_r4 <= last_beat_r3;
            last_beat_r5 <= last_beat_r4;
        end
    end

    // =========================================================================
    // compute_core (256-bit, original rtl/compute_core.sv)
    // =========================================================================
    logic signed [31:0] accum_out;
    logic               accum_clear;

    always_ff @(posedge clk)
        accum_clear <= rst ? 1'b0 : last_beat_r5;

    compute_core #(.VECTOR_WIDTH(VECTOR_WIDTH)) u_core (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (beat_valid_r2),
        .s_ready    (),
        .accum_clear(accum_clear),
        .act_in     (act_r2),
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
            if (last_beat_r5 && (!result_valid || m_axis_tready)) begin
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
        else if (last_beat_r5)
            tile_count <= tile_count + 1'b1;
    end

endmodule

`default_nettype wire
