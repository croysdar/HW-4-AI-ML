// =============================================================================
// File   : project/m4/sram_1macro_experiment/tb_top_narrow.sv
// Module : tb_top
//
// Co-simulation testbench for the narrow-compute, single-SRAM-macro bnn_top.
//
// Adapted from project/m4/tb/tb_top.sv. Functional contract is identical to
// the 256-bit baseline: AXI activation beats in, signed tile-dot-product on
// the AXI master interface, sv_dot reference unchanged. Differences are
// purely mechanical:
//
//   - WEIGHT_DEPTH parameter is 256 SRAM rows = 32 logical 256-bit beats
//   - load_weight writes 8 × 32-bit SRAM rows per logical weight word
//   - w_data is 32-bit; w_addr is 8-bit
//   - Throughput is 8× slower per tile (one beat = 8 SRAM/compute cycles)
//   - Phase plan tightened so the address counter stays below 32 logical
//     beats; an extra reload precedes the backpressure phase
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_top;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH    = 256;           // AXI side width
    localparam int  CHUNK_WIDTH     = 32;            // SRAM / compute width
    localparam int  CHUNKS_PER_BEAT = 8;             // VECTOR_WIDTH / CHUNK_WIDTH
    localparam int  WEIGHT_DEPTH    = 256;           // SRAM rows
    localparam int  LOGICAL_DEPTH   = WEIGHT_DEPTH / CHUNKS_PER_BEAT; // 32
    localparam int  MAX_BEATS       = 5;
    localparam real CLK_PERIOD      = 3.333;         // ns — 300 MHz

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
    logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr;          // 8 bits
    logic [CHUNK_WIDTH-1:0]          w_data;          // 32 bits

    logic [3:0]  cfg_beats_per_tile;
    logic [15:0] tile_count;

    // ── DUT instantiation ─────────────────────────────────────────────────────
    bnn_top #(
        .VECTOR_WIDTH    (VECTOR_WIDTH),
        .CHUNK_WIDTH     (CHUNK_WIDTH),
        .CHUNKS_PER_BEAT (CHUNKS_PER_BEAT),
        .WEIGHT_DEPTH    (WEIGHT_DEPTH),
        .MAX_BEATS       (MAX_BEATS)
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

    // Reference shadow: 256-bit logical weight words (unchanged semantics).
    logic [VECTOR_WIDTH-1:0] ref_weights [0:LOGICAL_DEPTH-1];

    logic [MAX_BEATS-1:0][VECTOR_WIDTH-1:0] act_buf;

    function automatic logic [255:0] rand256();
        logic [255:0] v;
        for (int w = 0; w < 8; w++)
            v[w*32 +: 32] = $urandom();
        return v;
    endfunction

    // ── Reference dot-product model (unchanged) ───────────────────────────────
    function automatic logic signed [31:0] sv_dot(
        input int base_addr,    // in logical-beat units
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

    // ── Task: load one LOGICAL weight word (256-bit) as 8 × 32-bit chunks ─────
    task automatic load_weight(input int logical_addr, input logic [255:0] wval);
        int base = logical_addr * CHUNKS_PER_BEAT;
        ref_weights[logical_addr] = wval;
        for (int c = 0; c < CHUNKS_PER_BEAT; c++) begin
            @(posedge clk); #1;
            w_en   = 1'b1;
            w_addr = $clog2(WEIGHT_DEPTH)'(base + c);
            w_data = wval[c*CHUNK_WIDTH +: CHUNK_WIDTH];
        end
        @(posedge clk); #1;
        w_en = 1'b0;
    endtask

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

    task automatic read_result(output logic signed [31:0] result);
        m_axis_tready = 1'b1;
        do @(posedge clk); while (!m_axis_tvalid);
        result = m_axis_tdata;
        #1;
        m_axis_tready = 1'b0;
    endtask

    task automatic fill_acts(input int n_beats);
        for (int b = 0; b < n_beats; b++)
            act_buf[b] = rand256();
    endtask

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

    task automatic flush_result();
        if (m_axis_tvalid) begin
            m_axis_tready = 1'b1;
            do @(posedge clk); while (!m_axis_tvalid);
            #1; m_axis_tready = 1'b0;
        end
        while (m_axis_tvalid) @(posedge clk);
        #1;
    endtask

    task automatic run_tile_bp(
        input int    n_beats,
        input int    w_base,
        input string tag
    );
        logic signed [31:0] expected, got;
        flush_result();
        fill_acts(n_beats);
        expected = sv_dot(w_base, n_beats);
        for (int b = 0; b < n_beats; b++)
            send_beat(act_buf[b], (b == n_beats - 1) ? 1'b1 : 1'b0);
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

    // ── Helper: reset DUT (clears w_ptr) ──────────────────────────────────────
    task automatic do_reset();
        rst = 1'b1;
        repeat (4) @(posedge clk); #1;
        rst = 1'b0;
        repeat (2) @(posedge clk); #1;
    endtask

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        $dumpfile("project/m4/sram_1macro_experiment/cosim_run.vcd");
        $dumpvars(0, tb_top);

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

        do_reset();

        // ── Load all 32 logical weight words (= 256 SRAM rows) ────────────────
        $display("── Loading %0d logical weight words (= %0d SRAM rows) ──",
                 LOGICAL_DEPTH, WEIGHT_DEPTH);
        for (int i = 0; i < LOGICAL_DEPTH; i++)
            load_weight(i, rand256());
        $display("   Weight load complete.");
        repeat (2) @(posedge clk); #1;

        // Phase 1 — conv4 (5 beats), 3 tiles, w_base 0/5/10. After: w_ptr=120.
        $display("── Phase 1: conv4 (5 beats/tile), 3 tiles ──");
        cfg_beats_per_tile = 4'd5;
        repeat (2) @(posedge clk); #1;
        run_tile_check(5,  0, "conv4");
        run_tile_check(5,  5, "conv4");
        run_tile_check(5, 10, "conv4");

        // Phase 2 — conv2 (2 beats), 4 tiles, w_base 15..21. After: w_ptr=184.
        $display("── Phase 2: conv2 (2 beats/tile), 4 tiles ──");
        cfg_beats_per_tile = 4'd2;
        repeat (2) @(posedge clk); #1;
        run_tile_check(2, 15, "conv2");
        run_tile_check(2, 17, "conv2");
        run_tile_check(2, 19, "conv2");
        run_tile_check(2, 21, "conv2");

        // Phase 3 — conv3 (3 beats), 3 tiles, w_base 23..29. After: w_ptr=256.
        // One fewer tile than the 256-bit baseline to stay within the 32-word
        // SRAM capacity (29+3 = 32 logical beats = 256 SRAM rows = exactly full).
        $display("── Phase 3: conv3 (3 beats/tile), 3 tiles ──");
        cfg_beats_per_tile = 4'd3;
        repeat (2) @(posedge clk); #1;
        run_tile_check(3, 23, "conv3");
        run_tile_check(3, 26, "conv3");
        run_tile_check(3, 29, "conv3");

        // ── Reset between phases — clears w_ptr so backpressure phase
        // can re-use w_base=0..14 without wrap concerns. Weights are SRAM-
        // resident (no clear); ref_weights shadow is also retained.
        $display("── Reset (clears w_ptr; weights remain in SRAM) ──");
        do_reset();
        cfg_beats_per_tile = 4'd5;

        // Phase 4 — backpressure on conv4 tiles, w_base 0..10.
        $display("── Phase 4: conv4 backpressure, 3 tiles ──");
        repeat (2) @(posedge clk); #1;
        run_tile_bp(5,  0, "conv4");
        run_tile_bp(5,  5, "conv4");
        run_tile_bp(5, 10, "conv4");

        // Phase 5 — weight reload. Overwrite logical words 0..4 with all-ones,
        // reset, then run one 5-beat tile of all-ones activations.
        // Expected: popcount=256, dot=256/beat, 5 beats → 1280.
        $display("── Phase 5: weight reload (all-ones pattern) ──");
        do_reset();
        cfg_beats_per_tile = 4'd5;
        for (int i = 0; i < 5; i++)
            load_weight(i, {VECTOR_WIDTH{1'b1}});
        repeat (2) @(posedge clk); #1;

        begin
            logic signed [31:0] got, expected;
            for (int b = 0; b < 5; b++)
                act_buf[b] = {VECTOR_WIDTH{1'b1}};
            expected = sv_dot(0, 5);   // 1280
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
