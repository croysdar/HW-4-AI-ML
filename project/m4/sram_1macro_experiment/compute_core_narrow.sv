// =============================================================================
// File   : project/m4/sram_1macro_experiment/compute_core_narrow.sv
// Module : compute_core
//
// Narrow-width XNOR-Popcount compute core. Identical algorithm to the 256-bit
// original, parameterized to a 32-bit datapath to match the width of a single
// sky130_sram_1kbyte_1rw1r_32x256_8 macro.
//
// Operation per clock (when s_valid):
//   xnor_bits = ~(act_in ^ weight_in)             // 32-bit XNOR
//   popcount  = popcount32(xnor_bits)             // 0..32 → 6-bit
//   chunk_dot = 2*popcount - 32                   // signed in [-32, +32]
//   accum    += chunk_dot                          // 32-bit signed
//
// A full 256-bit BNN dot product is accumulated across 8 consecutive chunks
// (8 cycles). The top-level FSM asserts accum_clear at the start of each
// filter tile so the accumulator collects all chunks across all beats of the
// tile into one signed sum. This matches the math of the 256-bit version:
//   dot_256(a, w) = Σ_{c=0..7} (2·popcount(a_c ⊕̄ w_c) − 32)
// because Σ chunk contributions = full dot mapping for 256-bit vectors.
//
// 3-stage pipeline (latency 3, throughput 1/clk):
//   Stage 1: XNOR                  → xnor_reg
//   Stage 2: 32-bit popcount       → popcount_r
//   Stage 3: 2*pop − 32, accumulate
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module compute_core #(
    parameter int VECTOR_WIDTH = 32
) (
    input  logic                   clk,
    input  logic                   rst,

    input  logic                   s_valid,
    output logic                   s_ready,

    input  logic                   accum_clear,

    input  logic [VECTOR_WIDTH-1:0] act_in,
    input  logic [VECTOR_WIDTH-1:0] weight_in,

    output logic signed [31:0]     accum_out
);

    // ── Stage 1: XNOR ─────────────────────────────────────────────────────────
    logic [VECTOR_WIDTH-1:0] xnor_bits;
    assign xnor_bits = ~(act_in ^ weight_in);

    logic [VECTOR_WIDTH-1:0] xnor_reg;
    logic                    s_valid_r;
    logic                    accum_clear_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            xnor_reg      <= '0;
            s_valid_r     <= 1'b0;
            accum_clear_r <= 1'b0;
        end else begin
            if (s_valid & s_ready)
                xnor_reg  <= xnor_bits;
            s_valid_r     <= s_valid & s_ready;
            accum_clear_r <= accum_clear;
        end
    end

    // ── Stage 2: 32-bit popcount ──────────────────────────────────────────────
    localparam int POP_W = $clog2(VECTOR_WIDTH + 1);  // 6 for 32-bit

    logic [POP_W-1:0] popcount;
    always_comb begin
        popcount = '0;
        for (int i = 0; i < VECTOR_WIDTH; i++)
            popcount += xnor_reg[i];
    end

    logic [POP_W-1:0] popcount_r;
    logic             s_valid_r2;
    logic             accum_clear_r2;

    always_ff @(posedge clk) begin
        if (rst) begin
            popcount_r     <= '0;
            s_valid_r2     <= 1'b0;
            accum_clear_r2 <= 1'b0;
        end else begin
            if (s_valid_r)
                popcount_r <= popcount;
            s_valid_r2     <= s_valid_r;
            accum_clear_r2 <= accum_clear_r;
        end
    end

    // ── Stage 3: chunk_dot, accumulate ────────────────────────────────────────
    logic signed [31:0] chunk_dot;
    assign chunk_dot = (32'(popcount_r) << 1) - 32'(VECTOR_WIDTH);

    always_ff @(posedge clk) begin
        if (rst)
            accum_out <= 32'sd0;
        else if (accum_clear_r2)
            accum_out <= 32'sd0;
        else if (s_valid_r2)
            accum_out <= accum_out + chunk_dot;
    end

    assign s_ready = 1'b1;

endmodule

`default_nettype wire
