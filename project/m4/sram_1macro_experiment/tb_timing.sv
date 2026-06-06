// =============================================================================
// File   : project/m4/sram_1macro_experiment/tb_timing.sv
// Module : tb_timing
//
// Cycle-accurate throughput benchmark for the narrow-compute bnn_top.
// Measures per-tile latency and full-frame inference time for conv2/conv3/conv4
// under back-to-back (no AXI stall) conditions — best-case chip throughput.
//
// Network dimensions (bnn_serengeti2.py):
//   conv2 : 112×112 output → 12544 tiles, 2 beats/tile, 32 ch in, 64 ch out
//   conv3 :  56× 56 output →  3136 tiles, 3 beats/tile, 64 ch in, 128 ch out
//   conv4 :  28× 28 output →   784 tiles, 5 beats/tile, 128 ch in, 256 ch out
//
// Throughput model per tile (no stall):
//   cycles = n_beats × CHUNKS_PER_BEAT + pipeline_drain
//   pipeline_drain = 4  (1 SRAM read + 3 compute stages + result latch)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_timing;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH    = 256;
    localparam int  CHUNK_WIDTH     = 32;
    localparam int  CHUNKS_PER_BEAT = 8;
    localparam int  WEIGHT_DEPTH    = 256;
    localparam int  LOGICAL_DEPTH   = WEIGHT_DEPTH / CHUNKS_PER_BEAT; // 32
    localparam int  MAX_BEATS       = 5;
    localparam real CLK_PERIOD_NS   = 50.0;   // 20 MHz (same as P&R target)
    localparam real CLK_FREQ_MHZ    = 1000.0 / CLK_PERIOD_NS;

    // Network layer tile counts
    localparam int CONV2_TILES  = 12544;
    localparam int CONV3_TILES  =  3136;
    localparam int CONV4_TILES  =   784;
    localparam int CONV2_BEATS  = 2;
    localparam int CONV3_BEATS  = 3;
    localparam int CONV4_BEATS  = 5;

    // ── DUT ports ─────────────────────────────────────────────────────────────
    logic        clk, rst;
    logic        s_axis_tvalid, s_axis_tready;
    logic [VECTOR_WIDTH-1:0] s_axis_tdata;
    logic        s_axis_tlast;
    logic        m_axis_tvalid, m_axis_tready;
    logic [31:0] m_axis_tdata;
    logic        m_axis_tlast;
    logic        w_en;
    logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr;
    logic [CHUNK_WIDTH-1:0]          w_data;
    logic [3:0]  cfg_beats_per_tile;
    logic [15:0] tile_count;

    bnn_top #(
        .VECTOR_WIDTH    (VECTOR_WIDTH),
        .CHUNK_WIDTH     (CHUNK_WIDTH),
        .CHUNKS_PER_BEAT (CHUNKS_PER_BEAT),
        .WEIGHT_DEPTH    (WEIGHT_DEPTH),
        .MAX_BEATS       (MAX_BEATS)
    ) dut (
        .clk               (clk),   .rst               (rst),
        .s_axis_tvalid     (s_axis_tvalid),
        .s_axis_tready     (s_axis_tready),
        .s_axis_tdata      (s_axis_tdata),
        .s_axis_tlast      (s_axis_tlast),
        .m_axis_tvalid     (m_axis_tvalid),
        .m_axis_tready     (m_axis_tready),
        .m_axis_tdata      (m_axis_tdata),
        .m_axis_tlast      (m_axis_tlast),
        .w_en              (w_en),  .w_addr (w_addr), .w_data (w_data),
        .cfg_beats_per_tile(cfg_beats_per_tile),
        .tile_count        (tile_count)
    );

    initial clk = 0;
    always #(CLK_PERIOD_NS / 2.0) clk = ~clk;

    // ── Weight load (fill SRAM with random data) ───────────────────────────
    task automatic load_all_weights();
        for (int i = 0; i < LOGICAL_DEPTH; i++) begin
            int base = i * CHUNKS_PER_BEAT;
            for (int c = 0; c < CHUNKS_PER_BEAT; c++) begin
                @(posedge clk); #1;
                w_en   = 1'b1;
                w_addr = ($clog2(WEIGHT_DEPTH))'(base + c);
                w_data = $urandom();
            end
        end
        @(posedge clk); #1;
        w_en = 1'b0;
    endtask

    // ── Send one beat (no stall on sender side) ────────────────────────────
    task automatic send_beat(input logic last);
        s_axis_tdata  = {$urandom(),$urandom(),$urandom(),$urandom(),
                         $urandom(),$urandom(),$urandom(),$urandom()};
        s_axis_tlast  = last;
        s_axis_tvalid = 1'b1;
        do @(posedge clk); while (!s_axis_tready);
        #1;
        s_axis_tvalid = 1'b0;
        s_axis_tlast  = 1'b0;
    endtask

    // ── Drain result output ────────────────────────────────────────────────
    task automatic drain_result();
        m_axis_tready = 1'b1;
        do @(posedge clk); while (!m_axis_tvalid);
        #1;
        m_axis_tready = 1'b0;
    endtask

    // ── Time N tiles back-to-back, return elapsed cycles ──────────────────
    task automatic time_layer(
        input  int    n_tiles,
        input  int    n_beats,
        output longint cycles_out
    );
        longint t_start, t_end;
        t_start = $time;
        for (int t = 0; t < n_tiles; t++) begin
            for (int b = 0; b < n_beats; b++)
                send_beat(b == n_beats-1);
            drain_result();
        end
        t_end = $time;
        cycles_out = (t_end - t_start) / longint'(CLK_PERIOD_NS);
    endtask

    // ── Main ──────────────────────────────────────────────────────────────
    initial begin
        rst = 1; s_axis_tvalid = 0; s_axis_tdata = 0; s_axis_tlast = 0;
        m_axis_tready = 0; w_en = 0; w_addr = 0; w_data = 0;
        cfg_beats_per_tile = 5;
        repeat (4) @(posedge clk); #1;
        rst = 0;
        repeat (2) @(posedge clk); #1;

        load_all_weights();
        repeat (2) @(posedge clk); #1;

        // ── Single-tile latency measurements ──────────────────────────────
        begin
            longint c2, c3, c4;

            $display("");
            $display("=== Single-tile latency (back-to-back, no stall) ===");

            cfg_beats_per_tile = CONV2_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV2_BEATS, c2);
            $display("  conv2 (2 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c2, c2*CLK_PERIOD_NS, CLK_FREQ_MHZ);

            // reset w_ptr for conv3
            rst = 1; repeat(4) @(posedge clk); #1; rst = 0; repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV3_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV3_BEATS, c3);
            $display("  conv3 (3 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c3, c3*CLK_PERIOD_NS, CLK_FREQ_MHZ);

            rst = 1; repeat(4) @(posedge clk); #1; rst = 0; repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV4_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV4_BEATS, c4);
            $display("  conv4 (5 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c4, c4*CLK_PERIOD_NS, CLK_FREQ_MHZ);
        end

        // ── Full-frame throughput (all tiles, back-to-back) ────────────────
        // Weight SRAM only holds 32 logical beats (6 conv4 filters resident).
        // Real chip reloads between filter groups; sim approximates by cycling
        // through the same 32 weights — throughput is identical either way
        // since weight reload overlaps with host-side processing.
        begin
            longint c2_total, c3_total, c4_total, frame_total;
            real    t2_us, t3_us, t4_us, frame_ms;
            real    energy_uj;
            localparam real POWER_MW = 2.91;  // post-PnR OpenSTA @ nom_tt_025C_1v80

            $display("");
            $display("=== Full-frame throughput (%0d conv2 + %0d conv3 + %0d conv4 tiles) ===",
                     CONV2_TILES, CONV3_TILES, CONV4_TILES);

            rst = 1; repeat(4) @(posedge clk); #1; rst = 0; repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV2_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV2_TILES, CONV2_BEATS, c2_total);
            t2_us = (c2_total * CLK_PERIOD_NS) / 1000.0;
            $display("  conv2: %0d tiles × %0d beats = %0d cycles  (%.1f µs)",
                     CONV2_TILES, CONV2_BEATS, c2_total, t2_us);

            rst = 1; repeat(4) @(posedge clk); #1; rst = 0; repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV3_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV3_TILES, CONV3_BEATS, c3_total);
            t3_us = (c3_total * CLK_PERIOD_NS) / 1000.0;
            $display("  conv3: %0d tiles × %0d beats = %0d cycles  (%.1f µs)",
                     CONV3_TILES, CONV3_BEATS, c3_total, t3_us);

            rst = 1; repeat(4) @(posedge clk); #1; rst = 0; repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV4_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV4_TILES, CONV4_BEATS, c4_total);
            t4_us = (c4_total * CLK_PERIOD_NS) / 1000.0;
            $display("  conv4: %0d tiles × %0d beats = %0d cycles  (%.1f µs)",
                     CONV4_TILES, CONV4_BEATS, c4_total, t4_us);

            frame_total = c2_total + c3_total + c4_total;
            frame_ms    = (frame_total * CLK_PERIOD_NS) / 1_000_000.0;
            energy_uj   = POWER_MW * frame_ms;  // mW × ms = µJ

            $display("");
            $display("─────────────────────────────────────────────────────────");
            $display("  Total BNN layers  : %0d cycles", frame_total);
            $display("  Frame latency     : %.2f ms  (budget: 33.3 ms @ 30 FPS)", frame_ms);
            $display("  Frame budget used : %.1f%%", (frame_ms / 33.33) * 100.0);
            $display("  Energy/frame      : %.2f µJ  (%.2f mW × %.2f ms)",
                     energy_uj, POWER_MW, frame_ms);
            $display("  Throughput        : %.1f FPS (BNN layers only)",
                     1000.0 / frame_ms);
            $display("─────────────────────────────────────────────────────────");
        end

        $finish;
    end

endmodule

`default_nettype wire
