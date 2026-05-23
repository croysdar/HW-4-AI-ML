// =============================================================================
// File   : project/m3/tb/tb_top.sv
// Module : tb_top
//
// End-to-end co-simulation testbench for bnn_top.
//
// Tests the dominant kernel from M1 profiling: conv4 (128 input channels,
// 3×3 kernel → 5 × 256-bit weight beats per filter tile).  Also exercises
// conv2 (2 beats) and conv3 (3 beats) by reconfiguring cfg_beats_per_tile
// between phases.
//
// Protocol:
//   All input data enters exclusively through the AXI4-Stream slave ports
//   (s_axis_*).  All results are read exclusively through the AXI4-Stream
//   master ports (m_axis_*).  The testbench never touches internal DUT
//   signals for data transfer — only the interfaces a real host would use.
//
// Reference model:
//   An independent SystemVerilog function (sv_dot) computes the expected
//   accumulator value using $countones on the XNOR of each activation beat
//   with the corresponding weight word.  This is entirely separate from the
//   DUT's for-loop popcount.
//
// Test phases:
//   Phase 1 — conv4 tile (5 beats): 3 tiles, random weights + activations.
//   Phase 2 — conv2 tile (2 beats): 4 tiles, cfg reconfiguration.
//   Phase 3 — conv3 tile (3 beats): 4 tiles.
//   Phase 4 — backpressure: conv4 tiles with random m_axis_tready deassertions.
//   Phase 5 — weight reload: reload weights and verify result changes.
//
// PASS/FAIL: single line printed at end. Grader reads the log.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_top;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH  = 256;
    localparam int  WEIGHT_DEPTH  = 1512;
    localparam int  MAX_BEATS     = 5;
    localparam real CLK_PERIOD    = 3.333; // ns — 300 MHz

    // ── DUT ports ─────────────────────────────────────────────────────────────
    logic        clk, rst;

    logic        s_axis_tvalid;
    logic        s_axis_tready;
    logic [VECTOR_WIDTH-1:0] s_axis_tdata;
    logic        s_axis_tlast;

    logic        m_axis_tvalid;
    logic        m_axis_tready;
    logic [31:0] m_axis_tdata;
    logic        m_axis_tlast;

    logic        w_en;
    logic [10:0] w_addr;
    logic [VECTOR_WIDTH-1:0] w_data;

    logic [3:0]  cfg_beats_per_tile;
    logic [15:0] tile_count;

    // ── DUT instantiation ─────────────────────────────────────────────────────
    bnn_top #(
        .VECTOR_WIDTH(VECTOR_WIDTH),
        .WEIGHT_DEPTH(WEIGHT_DEPTH),
        .MAX_BEATS   (MAX_BEATS)
    ) dut (
        .clk               (clk),
        .rst               (rst),
        .s_axis_tvalid     (s_axis_tvalid),
        .s_axis_tready     (s_axis_tready),
        .s_axis_tdata      (s_axis_tdata),
        .s_axis_tlast      (s_axis_tlast),
        .m_axis_tvalid     (m_axis_tvalid),
        .m_axis_tready     (m_axis_tready),
        .m_axis_tdata      (m_axis_tdata),
        .m_axis_tlast      (m_axis_tlast),
        .w_en              (w_en),
        .w_addr            (w_addr),
        .w_data            (w_data),
        .cfg_beats_per_tile(cfg_beats_per_tile),
        .tile_count        (tile_count)
    );

    // ── Clock ─────────────────────────────────────────────────────────────────
    initial clk = 1'b0;
    always  #(CLK_PERIOD / 2.0) clk = ~clk;

    // ── Test state ────────────────────────────────────────────────────────────
    int fail_count;

    // Reference weight shadow — mirrors what was loaded into the DUT.
    logic [VECTOR_WIDTH-1:0] ref_weights [0:WEIGHT_DEPTH-1];

    // Shared activation buffer (written before each tile, read by sv_dot).
    // Packed as [beat][bit]: avoids unpacked-array-port limitation in Icarus.
    logic [MAX_BEATS-1:0][VECTOR_WIDTH-1:0] act_buf;

    // ── Helper: random 256-bit word ───────────────────────────────────────────
    function automatic logic [255:0] rand256();
        logic [255:0] v;
        for (int w = 0; w < 8; w++)
            v[w*32 +: 32] = $urandom();
        return v;
    endfunction

    // ── Reference dot-product model ───────────────────────────────────────────
    // Reads act_buf[0..n_beats-1] and ref_weights[base_addr..base_addr+n_beats-1].
    // Returns the expected signed accumulator value.
    function automatic logic signed [31:0] sv_dot(
        input int base_addr,
        input int n_beats
    );
        logic signed [31:0] acc;
        logic [VECTOR_WIDTH-1:0] xnor_bits;
        int pop;
        acc = 32'sd0;
        for (int b = 0; b < n_beats; b++) begin
            xnor_bits = ~(act_buf[b] ^ ref_weights[base_addr + b]);
            pop       = $countones(xnor_bits);
            acc       = acc + (pop * 2) - VECTOR_WIDTH;
        end
        return acc;
    endfunction

    // ── Task: load one weight word via sideband port ──────────────────────────
    task automatic load_weight(input int addr, input logic [255:0] wval);
        @(posedge clk); #1;
        w_en              = 1'b1;
        w_addr            = 11'(addr);
        w_data            = wval;
        ref_weights[addr] = wval;
        @(posedge clk); #1;
        w_en = 1'b0;
    endtask

    // ── Task: send one AXI beat through the slave interface ───────────────────
    task automatic send_beat(
        input logic [VECTOR_WIDTH-1:0] data,
        input logic                    tlast
    );
        s_axis_tdata  = data;
        s_axis_tlast  = tlast;
        s_axis_tvalid = 1'b1;
        do @(posedge clk); while (!s_axis_tready);
        #1;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
    endtask

    // ── Task: read one result from the AXI master interface ───────────────────
    task automatic read_result(output logic signed [31:0] result);
        m_axis_tready = 1'b1;
        do @(posedge clk); while (!m_axis_tvalid);
        result = m_axis_tdata;
        #1;
        m_axis_tready = 1'b0;
    endtask

    // ── Task: fill act_buf with random data ───────────────────────────────────
    task automatic fill_acts(input int n_beats);
        for (int b = 0; b < n_beats; b++)
            act_buf[b] = rand256();
    endtask

    // ── Task: run one tile and check result ───────────────────────────────────
    task automatic run_tile_check(
        input int    n_beats,
        input int    w_base,
        input string tag
    );
        logic signed [31:0] expected, got;

        fill_acts(n_beats);
        expected = sv_dot(w_base, n_beats);

        for (int b = 0; b < n_beats; b++)
            send_beat(act_buf[b], (b == n_beats - 1) ? 1'b1 : 1'b0);

        read_result(got);

        if (got !== expected) begin
            $display("FAIL [%s] w_base=%0d | DUT=%0d expected=%0d",
                     tag, w_base, got, expected);
            fail_count++;
        end else begin
            $display("  OK  [%s] w_base=%0d | result=%0d", tag, w_base, got);
        end
    endtask

    // ── Task: flush any pending result from the master interface ──────────────
    // Ensures result_valid is low before starting a new backpressure tile,
    // so the receive thread cannot accidentally capture a stale result.
    task automatic flush_result();
        if (m_axis_tvalid) begin
            m_axis_tready = 1'b1;
            do @(posedge clk); while (!m_axis_tvalid);
            #1; m_axis_tready = 1'b0;
        end
        // Also wait until valid deasserts
        while (m_axis_tvalid) @(posedge clk);
        #1;
    endtask

    // ── Task: run one tile with backpressure on m_axis_tready ─────────────────
    task automatic run_tile_bp(
        input int    n_beats,
        input int    w_base,
        input string tag
    );
        logic signed [31:0] expected, got;

        // Flush any leftover result before starting so the recv thread
        // only sees the result produced by this tile's beats.
        flush_result();

        fill_acts(n_beats);
        expected = sv_dot(w_base, n_beats);

        // Stream beats, then wait for result with random backpressure.
        // Sequential (not fork) avoids racing against a stale result_valid.
        for (int b = 0; b < n_beats; b++)
            send_beat(act_buf[b], (b == n_beats - 1) ? 1'b1 : 1'b0);

        // Now randomly toggle tready until we capture the result.
        got = '0;
        forever begin
            @(posedge clk); #1;
            m_axis_tready = $urandom_range(0, 1);
            if (m_axis_tvalid && m_axis_tready) begin
                got = m_axis_tdata;
                m_axis_tready = 1'b0;
                break;
            end
        end

        if (got !== expected) begin
            $display("FAIL [%s bp] w_base=%0d | DUT=%0d expected=%0d",
                     tag, w_base, got, expected);
            fail_count++;
        end else begin
            $display("  OK  [%s bp] w_base=%0d | result=%0d", tag, w_base, got);
        end
    endtask

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        $dumpfile("project/m3/sim/cosim_run.vcd");
        $dumpvars(0, tb_top);

        // Initialise
        rst                = 1'b1;
        s_axis_tvalid      = 1'b0;
        s_axis_tdata       = '0;
        s_axis_tlast       = 1'b0;
        m_axis_tready      = 1'b0;
        w_en               = 1'b0;
        w_addr             = '0;
        w_data             = '0;
        cfg_beats_per_tile = 4'd5;
        fail_count         = 0;

        repeat (4) @(posedge clk);
        #1; rst = 1'b0;
        repeat (2) @(posedge clk); #1;

        // ── Load all weights via sideband port ────────────────────────────────
        $display("── Loading %0d weight words via sideband ──", WEIGHT_DEPTH);
        for (int i = 0; i < WEIGHT_DEPTH; i++)
            load_weight(i, rand256());
        $display("   Weight load complete.");
        repeat (2) @(posedge clk); #1;

        // ─────────────────────────────────────────────────────────────────────
        // Phase 1: conv4 (5 beats/tile) — dominant kernel from M1 profiling
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 1: conv4 (5 beats/tile), 3 tiles ──");
        cfg_beats_per_tile = 4'd5;
        repeat (2) @(posedge clk); #1;

        run_tile_check(5,  0, "conv4");
        run_tile_check(5,  5, "conv4");
        run_tile_check(5, 10, "conv4");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: conv2 (2 beats/tile)
        // w_ptr is at 15 after 3 conv4 tiles (3×5=15 beats consumed).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 2: conv2 (2 beats/tile), 4 tiles ──");
        cfg_beats_per_tile = 4'd2;
        repeat (2) @(posedge clk); #1;

        run_tile_check(2, 15, "conv2");
        run_tile_check(2, 17, "conv2");
        run_tile_check(2, 19, "conv2");
        run_tile_check(2, 21, "conv2");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: conv3 (3 beats/tile)
        // w_ptr is at 23 after Phase 2 (15 + 4×2 = 23).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 3: conv3 (3 beats/tile), 4 tiles ──");
        cfg_beats_per_tile = 4'd3;
        repeat (2) @(posedge clk); #1;

        run_tile_check(3, 23, "conv3");
        run_tile_check(3, 26, "conv3");
        run_tile_check(3, 29, "conv3");
        run_tile_check(3, 32, "conv3");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4: conv4 with random m_axis_tready backpressure
        // w_ptr is at 35 after Phase 3 (23 + 4×3 = 35).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 4: conv4 backpressure, 3 tiles ──");
        cfg_beats_per_tile = 4'd5;
        repeat (2) @(posedge clk); #1;

        run_tile_bp(5, 35, "conv4");
        run_tile_bp(5, 40, "conv4");
        run_tile_bp(5, 45, "conv4");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 5: weight reload — overwrite words 0-4, reset, verify result
        // Expected: all-ones act XOR all-ones weight = all-ones XNOR = all-ones
        //           popcount = 256, dot = 256 per beat, 5 beats → total = 1280
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 5: weight reload (all-ones pattern) ──");
        rst = 1'b1;
        repeat (4) @(posedge clk); #1;
        rst = 1'b0;
        repeat (2) @(posedge clk); #1;
        cfg_beats_per_tile = 4'd5;

        for (int i = 0; i < 5; i++)
            load_weight(i, {VECTOR_WIDTH{1'b1}});
        repeat (2) @(posedge clk); #1;

        begin
            logic signed [31:0] got, expected;
            for (int b = 0; b < 5; b++)
                act_buf[b] = {VECTOR_WIDTH{1'b1}};
            expected = sv_dot(0, 5);   // expects 1280

            for (int b = 0; b < 5; b++)
                send_beat(act_buf[b], (b == 4) ? 1'b1 : 1'b0);
            read_result(got);

            if (got !== expected) begin
                $display("FAIL [weight reload] DUT=%0d expected=%0d", got, expected);
                fail_count++;
            end else begin
                $display("  OK  [weight reload] result=%0d (expected %0d)", got, expected);
            end
        end

        // ── Verdict ───────────────────────────────────────────────────────────
        repeat (4) @(posedge clk);
        $display("─────────────────────────────────────────────────");
        if (fail_count == 0)
            $display("VERIFIABLE PASS — all tile checks matched sv_dot reference");
        else
            $display("FAIL — %0d mismatches detected", fail_count);
        $display("─────────────────────────────────────────────────");

        $finish;
    end

endmodule

`default_nettype wire
