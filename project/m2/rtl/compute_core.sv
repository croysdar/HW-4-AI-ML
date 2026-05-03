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

    // ── Combinational: XNOR → Popcount → dot_val ──────────────────────────────
    logic [VECTOR_WIDTH-1:0]          xnor_bits;
    logic [$clog2(VECTOR_WIDTH+1)-1:0] popcount;
    logic signed [31:0]               dot_val;

    assign xnor_bits = ~(act_in ^ weight_in);

    // Synthesizable for-loop: Yosys unrolls this into a balanced adder tree.
    always_comb begin
        popcount = '0;
        for (int i = 0; i < VECTOR_WIDTH; i++)
            popcount += xnor_bits[i];
    end

    // Map [0, N] → [−N, +N]:  dot = 2·popcount − VECTOR_WIDTH
    assign dot_val = (32'(signed'(popcount)) << 1) - 32'(VECTOR_WIDTH);

    // ── Sequential: Accumulator ───────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst)
            accum_out <= 32'sd0;
        else if (accum_clear)
            accum_out <= 32'sd0;
        else if (s_valid && s_ready)
            accum_out <= accum_out + dot_val;
    end

    // No backpressure — always ready to accept a new vector.
    assign s_ready = 1'b1;

endmodule

`default_nettype wire
