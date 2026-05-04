// tb_compute_core.sv — Self-Checking Testbench for compute_core
// ============================================================
// Verifiable PASS strategy:
//   DUT uses a combinational for-loop to compute popcount.
//   Testbench uses SystemVerilog's built-in $countones() — a completely
//   independent reference — to compute the expected accumulator value.
//   After every valid handshake, the TB asserts DUT accum_out == expected.
//   Mismatch on any cycle prints FAIL and terminates; all-pass prints
//   VERIFIABLE PASS.
//
// Tests covered:
//   1. N_VECS random vectors with continuous valid — accumulation correctness.
//   2. Interleaved s_valid=0 cycles  — accumulator must hold value.
//   3. accum_clear mid-stream        — accumulator must reset to 0.
//   4. Synchronous rst mid-stream    — accumulator must reset to 0.

`timescale 1ns/1ps
`default_nettype none

module tb_compute_core;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH = 256;
    localparam int  N_VECS       = 32;   // random vectors per phase
    localparam real CLK_PERIOD   = 10.0; // ns (100 MHz — convenient for simulation)

    // ── DUT port signals ──────────────────────────────────────────────────────
    logic                     clk;
    logic                     rst;
    logic                     s_valid;
    logic                     s_ready;
    logic                     accum_clear;
    logic [VECTOR_WIDTH-1:0]  act_in;
    logic [VECTOR_WIDTH-1:0]  weight_in;
    logic signed [31:0]       accum_out;

    // ── DUT instantiation ─────────────────────────────────────────────────────
    compute_core #(
        .VECTOR_WIDTH(VECTOR_WIDTH)
    ) dut (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (s_valid),
        .s_ready    (s_ready),
        .accum_clear(accum_clear),
        .act_in     (act_in),
        .weight_in  (weight_in),
        .accum_out  (accum_out)
    );

    // ── Clock generation ──────────────────────────────────────────────────────
    initial clk = 1'b0;
    always #(CLK_PERIOD / 2.0) clk = ~clk;

    // ── Test state ────────────────────────────────────────────────────────────
    int          fail_count;
    logic signed [31:0] expected_accum;

    // ── Tasks ─────────────────────────────────────────────────────────────────

    // Apply one vector and check result
    task automatic apply_and_check(
        input logic [VECTOR_WIDTH-1:0] act,
        input logic [VECTOR_WIDTH-1:0] wt
    );
        logic [VECTOR_WIDTH-1:0] xnor_ref;
        int                      pop_ref;
        logic signed [31:0]      dot_ref;

        act_in    = act;
        weight_in = wt;
        s_valid   = 1'b1;

        // Independent reference: $countones on the XNOR of act and weight.
        // This path is entirely separate from the DUT's for-loop popcount.
        xnor_ref = ~(act ^ wt);
        pop_ref  = $countones(xnor_ref);
        dot_ref  = (pop_ref * 2) - VECTOR_WIDTH;
        expected_accum = expected_accum + dot_ref;

        @(posedge clk);  // DUT samples on this edge (s_valid && s_ready)
        #1;              // small propagation delay before sampling output

        if (accum_out !== expected_accum) begin
            $display("FAIL: act=%0h wt=%0h | DUT accum=%0d  expected=%0d",
                     act, wt, accum_out, expected_accum);
            fail_count++;
        end
    endtask

    // Hold s_valid low for N cycles and confirm accum_out does not change
    task automatic hold_check(input int n_cycles);
        logic signed [31:0] snap;
        s_valid = 1'b0;
        snap    = expected_accum;  // no change expected
        repeat (n_cycles) begin
            @(posedge clk);
            #1;
            if (accum_out !== snap) begin
                $display("FAIL: accum changed during s_valid=0: DUT=%0d expected=%0d",
                         accum_out, snap);
                fail_count++;
            end
        end
    endtask

    // ── Random vector generator (256-bit from 8 × 32-bit $urandom calls) ────
    // $urandom needs no seed variable; state is seeded once at simulation start.
    function automatic logic [255:0] rand256();
        logic [255:0] v;
        for (int w = 0; w < 8; w++)
            v[w*32 +: 32] = $urandom();
        return v;
    endfunction

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        $dumpfile("project/m2/sim/compute_core.vcd");
        $dumpvars(0, tb_compute_core);

        // Initialise all inputs
        rst          = 1'b1;
        s_valid      = 1'b0;
        accum_clear  = 1'b0;
        act_in       = '0;
        weight_in    = '0;
        fail_count   = 0;
        expected_accum = 32'sd0;

        // ── Phase 0: synchronous reset ────────────────────────────────────────
        @(posedge clk); @(posedge clk);
        rst = 1'b0;
        @(posedge clk); #1;

        $display("── Phase 1: %0d random vectors, continuous valid ──", N_VECS);
        for (int i = 0; i < N_VECS; i++) begin
            apply_and_check(rand256(), rand256());
        end
        s_valid = 1'b0;

        // ── Phase 2: interleaved idle cycles ──────────────────────────────────
        $display("── Phase 2: 4 idle cycles — accumulator must hold ──");
        hold_check(4);

        $display("── Phase 3: 8 more vectors after idle ──");
        for (int i = N_VECS; i < N_VECS + 8; i++) begin
            apply_and_check(rand256(), rand256());
            if (i % 3 == 0)
                hold_check(2);  // stall every third beat
        end
        s_valid = 1'b0;

        // ── Phase 4: accum_clear mid-stream ───────────────────────────────────
        $display("── Phase 4: accum_clear — must zero accumulator ──");
        // Send one more vector to ensure accum_out is non-zero
        apply_and_check(rand256(), rand256());
        s_valid = 1'b0;

        accum_clear    = 1'b1;
        expected_accum = 32'sd0;
        @(posedge clk); #1;
        accum_clear = 1'b0;

        if (accum_out !== 32'sd0) begin
            $display("FAIL: accum_clear did not zero DUT: accum_out=%0d", accum_out);
            fail_count++;
        end else begin
            $display("  accum_clear OK: accum_out=%0d (expected 0)", accum_out);
        end

        // ── Phase 5: 8 vectors after clear ────────────────────────────────────
        $display("── Phase 5: 8 vectors after accum_clear ──");
        for (int i = 0; i < 8; i++) begin
            apply_and_check(rand256(), rand256());
        end
        s_valid = 1'b0;

        // ── Phase 6: synchronous rst mid-stream ───────────────────────────────
        $display("── Phase 6: synchronous rst — must zero accumulator ──");
        apply_and_check(rand256(), rand256());
        s_valid = 1'b0;

        rst            = 1'b1;
        expected_accum = 32'sd0;
        @(posedge clk); #1;
        rst = 1'b0;

        if (accum_out !== 32'sd0) begin
            $display("FAIL: rst did not zero DUT: accum_out=%0d", accum_out);
            fail_count++;
        end else begin
            $display("  rst OK: accum_out=%0d (expected 0)", accum_out);
        end

        // ── Verdict ───────────────────────────────────────────────────────────
        $display("─────────────────────────────────────────────────");
        if (fail_count == 0)
            $display("VERIFIABLE PASS — all vectors matched $countones reference");
        else
            $display("FAIL — %0d mismatches detected", fail_count);
        $display("─────────────────────────────────────────────────");

        $finish;
    end

endmodule

`default_nettype wire
