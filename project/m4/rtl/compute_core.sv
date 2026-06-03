// =============================================================================
// File   : project/m2/rtl/compute_core.sv
// Module : compute_core
//
// Description
// -----------
// 1-bit XNOR-Popcount compute core for the BNN hardware accelerator chiplet.
// Maps to the chiplet partition: BinarizeConv2d layers conv2, conv3, conv4.
// Conv1 runs as INT8 on the ARM host CPU; only the 1-bit binary layers execute
// here.  See project/m1/interface_selection.md for bus sizing rationale.
//
// On each valid handshake the core computes:
//   xnor_bits = ~(act_in XOR weight_in)          // 256-bit bitwise XNOR
//   popcount  = sum of 1-bits in xnor_bits        // Yosys adder tree
//   dot_val   = 2 * popcount − VECTOR_WIDTH       // maps [0,N] → [−N,+N]
//   accum_out += dot_val                          // 32-bit signed accumulator
//
// Pipeline
// --------
// 3-stage pipeline (latency 3 cycles, throughput 1/clk):
//   Stage 1: XNOR                  → xnor_reg
//   Stage 2: per-chunk popcount    → chunk_sums_r  (8 × 32-bit popcounts)
//   Stage 3: final sum + accumulate
//
// Clock / Reset
// -------------
// Single clock domain.  Target frequency: 300 MHz (same domain as AXI4-Stream).
// Reset: rst — active-HIGH, SYNCHRONOUS (sampled on posedge clk).
//
// Accumulator Update Priority (highest → lowest)
// -----------------------------------------------
//   1. rst         asserted → clear accum_out to 0
//   2. accum_clear asserted → clear accum_out to 0  (start of new filter tile)
//   3. s_valid & s_ready    → accum_out += dot_val
//   4. (none)               → hold current value
//
// Port Descriptions
// -----------------
// clk         in   1      System clock, rising-edge active.
// rst         in   1      Active-high synchronous reset. Clears accumulator.
// s_valid     in   1      AXI4-Stream-style: input vector is valid this cycle.
// s_ready     out  1      Core is always ready (no backpressure). Tied high.
// accum_clear in   1      Synchronous accumulator clear (start of new tile).
// act_in      in   256    Packed 1-bit activation vector from axis_interface.
// weight_in   in   256    Packed 1-bit weight vector (held stable per tile).
// accum_out   out  32(s)  Running signed dot-product accumulator output.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module compute_core #(
    parameter int VECTOR_WIDTH = 256
) (
    input  logic                   clk,
    input  logic                   rst,          // active-high synchronous reset

    // AXI4-Stream-style handshake
    input  logic                   s_valid,
    output logic                   s_ready,

    // Accumulator clear (start of new filter window)
    input  logic                   accum_clear,

    // 1-bit packed vectors
    input  logic [VECTOR_WIDTH-1:0] act_in,
    input  logic [VECTOR_WIDTH-1:0] weight_in,

    // Signed 32-bit running dot-product accumulator
    output logic signed [31:0]     accum_out
);

    // ── Stage 1 (combinational): XNOR ─────────────────────────────────────────
    // Kept purely combinational; registered at end of stage to cut timing path.
    logic [VECTOR_WIDTH-1:0] xnor_bits;
    assign xnor_bits = ~(act_in ^ weight_in);

    // ── Pipeline register 1: XNOR → chunk popcount stage ──────────────────────
    // Captures bit-wise XNOR result. Latency 1 cycle, throughput 1/clk.
    logic [VECTOR_WIDTH-1:0] xnor_reg;
    logic                    s_valid_r;
    logic                    accum_clear_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            xnor_reg      <= '0;
            s_valid_r     <= 1'b0;
            accum_clear_r <= 1'b0;
        end else begin
            xnor_reg      <= xnor_bits;
            s_valid_r     <= s_valid & s_ready;
            accum_clear_r <= accum_clear;
        end
    end

    // ── Stage 2 (combinational): per-chunk popcount ───────────────────────────
    // Split the 256-bit popcount tree into 8 parallel 32-bit popcounts.
    // Each chunk's max value is 32 → fits in 6 bits. This bounds the per-stage
    // adder depth: 32→6-bit popcount is ~5 levels, well under 3.33 ns even with
    // wire load. Stage 3 then sums 8×6-bit values (~3 levels).
    localparam int CHUNK_WIDTH = 32;
    localparam int N_CHUNKS    = VECTOR_WIDTH / CHUNK_WIDTH;  // 8
    localparam int CHUNK_SUM_W = $clog2(CHUNK_WIDTH + 1);     // 6

    logic [N_CHUNKS-1:0][CHUNK_SUM_W-1:0] chunk_sums;

    always_comb begin
        for (int c = 0; c < N_CHUNKS; c++) begin
            logic [CHUNK_SUM_W-1:0] s;
            s = '0;
            for (int i = 0; i < CHUNK_WIDTH; i++)
                s += xnor_reg[c*CHUNK_WIDTH + i];
            chunk_sums[c] = s;
        end
    end

    // ── Pipeline register 2: chunk sums + control ─────────────────────────────
    // Splits the Stage 2 adder tree from the Stage 3 final sum + accumulate.
    // Latency now 3 cycles; throughput unchanged at 1/clk.
    logic [N_CHUNKS-1:0][CHUNK_SUM_W-1:0] chunk_sums_r;
    logic                                  s_valid_r2;
    logic                                  accum_clear_r2;

    always_ff @(posedge clk) begin
        if (rst) begin
            chunk_sums_r   <= '0;
            s_valid_r2     <= 1'b0;
            accum_clear_r2 <= 1'b0;
        end else begin
            chunk_sums_r   <= chunk_sums;
            s_valid_r2     <= s_valid_r;
            accum_clear_r2 <= accum_clear_r;
        end
    end

    // ── Stage 3 (combinational): final sum + dot_val ──────────────────────────
    logic [$clog2(VECTOR_WIDTH+1)-1:0] popcount;
    logic signed [31:0]                dot_val;

    always_comb begin
        popcount = '0;
        for (int c = 0; c < N_CHUNKS; c++)
            popcount += chunk_sums_r[c];
    end

    // Map [0, N] → [−N, +N]:  dot = 2·popcount − VECTOR_WIDTH
    // Zero-extend popcount before signed arithmetic; signed'() on a 9-bit value
    // would misinterpret popcount=256 (max for 256-wide input) as negative.
    assign dot_val = (32'(popcount) << 1) - 32'(VECTOR_WIDTH);

    // ── Stage 3 (sequential): Accumulator ────────────────────────────────────
    // Uses twice-pipelined valid/clear so control signals stay aligned with data.
    always_ff @(posedge clk) begin
        if (rst)
            accum_out <= 32'sd0;
        else if (accum_clear_r2)
            accum_out <= 32'sd0;
        else if (s_valid_r2)
            accum_out <= accum_out + dot_val;
    end

    // No backpressure — always ready to accept a new vector.
    assign s_ready = 1'b1;

endmodule

`default_nettype wire
