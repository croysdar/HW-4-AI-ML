// =============================================================================
// File   : project/m4/tb/tb_top.sv
// Module : tb_top
//
// End-to-end co-simulation testbench for bnn_top (4-SRAM-macro final design).
//
// Key differences from the register-file testbench:
//   - WEIGHT_DEPTH = 128 (4 banks × 256 rows / 2 phases = 128 logical 256-bit words)
//   - w_data is 32-bit; load_weight sends 8 consecutive w_en cycles per word
//     (DUT's w_bank_sel auto-increments to route each 32-bit chunk to the
//      correct bank and half)
//   - Clock period 25 ns (40 MHz) to match the 4-SRAM macro timing target
//   - 5-cycle pipeline latency (2 SRAM phases + 3 compute stages)
//
// Test phases mirror the register-file testbench exactly so that results are
// directly comparable.  All w_base addresses fit within WEIGHT_DEPTH=128.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_top;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH  = 256;
    localparam int  WEIGHT_DEPTH  = 128;   // logical 256-bit words in 4-SRAM design
    localparam int  MAX_BEATS     = 5;
    localparam real CLK_PERIOD    = 25.0;  // ns — 40 MHz

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

    // 4-SRAM write interface: 32-bit wide, 8 cycles per 256-bit word
    logic        w_en;
    logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr;
    logic [31:0] w_data;

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

    // Reference weight shadow — full 256-bit words mirroring DUT SRAM contents.
    logic [VECTOR_WIDTH-1:0] ref_weights [0:WEIGHT_DEPTH-1];

    // Shared activation buffer.
    logic [MAX_BEATS-1:0][VECTOR_WIDTH-1:0] act_buf;

    // ── Helper: random 256-bit word ───────────────────────────────────────────
    function automatic logic [255:0] rand256();
        logic [255:0] v;
        for (int w = 0; w < 8; w++)
            v[w*32 +: 32] = $urandom();
        return v;
    endfunction

    // ── Reference dot-product model ───────────────────────────────────────────
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

    // ── Task: load one 256-bit weight word via 8 × 32-bit w_en cycles ─────────
    // The DUT's w_bank_sel auto-increments each cycle, routing each 32-bit
    // chunk to the correct bank and SRAM half.  w_addr must be held stable.
    task automatic load_weight(input int addr, input logic [255:0] wval);
        for (int chunk = 0; chunk < 8; chunk++) begin
            @(posedge clk); #1;
            w_en   = 1'b1;
            w_addr = $clog2(WEIGHT_DEPTH)'(addr);
            w_data = wval[chunk*32 +: 32];
        end
        @(posedge clk); #1;
        w_en = 1'b0;
        ref_weights[addr] = wval;
    endtask

    // ── Task: send one AXI beat ───────────────────────────────────────────────
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

    // ── Task: read one result from AXI master ────────────────────────────────
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

    // ── Task: flush pending result ────────────────────────────────────────────
    task automatic flush_result();
        if (m_axis_tvalid) begin
            m_axis_tready = 1'b1;
            do @(posedge clk); while (!m_axis_tvalid);
            #1; m_axis_tready = 1'b0;
        end
        while (m_axis_tvalid) @(posedge clk);
        #1;
    endtask

    // ── Task: run one tile with random m_axis_tready backpressure ─────────────
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

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        $dumpfile("project/m4/sram_4macro_experiment/sim/cosim_run.vcd");
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

        // ── Load all weights (128 words × 8 cycles each) ──────────────────────
        $display("── Loading %0d weight words via sideband (8 cycles/word) ──", WEIGHT_DEPTH);
        for (int i = 0; i < WEIGHT_DEPTH; i++)
            load_weight(i, rand256());
        $display("   Weight load complete.");
        repeat (2) @(posedge clk); #1;

        // ─────────────────────────────────────────────────────────────────────
        // Phase 1: conv4 (5 beats/tile), 3 tiles
        // w_ptr starts at 0; advances to 15 after this phase.
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 1: conv4 (5 beats/tile), 3 tiles ──");
        cfg_beats_per_tile = 4'd5;
        repeat (2) @(posedge clk); #1;

        run_tile_check(5,  0, "conv4");
        run_tile_check(5,  5, "conv4");
        run_tile_check(5, 10, "conv4");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: conv2 (2 beats/tile), 4 tiles
        // w_ptr at 15 after Phase 1 (3×5=15 beats consumed).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 2: conv2 (2 beats/tile), 4 tiles ──");
        cfg_beats_per_tile = 4'd2;
        repeat (2) @(posedge clk); #1;

        run_tile_check(2, 15, "conv2");
        run_tile_check(2, 17, "conv2");
        run_tile_check(2, 19, "conv2");
        run_tile_check(2, 21, "conv2");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: conv3 (3 beats/tile), 4 tiles
        // w_ptr at 23 after Phase 2 (15 + 4×2 = 23).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 3: conv3 (3 beats/tile), 4 tiles ──");
        cfg_beats_per_tile = 4'd3;
        repeat (2) @(posedge clk); #1;

        run_tile_check(3, 23, "conv3");
        run_tile_check(3, 26, "conv3");
        run_tile_check(3, 29, "conv3");
        run_tile_check(3, 32, "conv3");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4: conv4 with random m_axis_tready backpressure, 3 tiles
        // w_ptr at 35 after Phase 3 (23 + 4×3 = 35).
        // ─────────────────────────────────────────────────────────────────────
        $display("── Phase 4: conv4 backpressure, 3 tiles ──");
        cfg_beats_per_tile = 4'd5;
        repeat (2) @(posedge clk); #1;

        run_tile_bp(5, 35, "conv4");
        run_tile_bp(5, 40, "conv4");
        run_tile_bp(5, 45, "conv4");

        // ─────────────────────────────────────────────────────────────────────
        // Phase 5: weight reload — overwrite words 0-4 with all-ones, verify
        // Expected: all-ones act XNOR all-ones weight = all-ones (256 bits)
        //           popcount=256, dot=256 per beat, 5 beats → total = 1280
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
