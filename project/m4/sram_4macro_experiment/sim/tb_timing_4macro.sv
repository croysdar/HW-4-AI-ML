// =============================================================================
// File   : project/m4/sram_4macro_experiment/sim/tb_timing_4macro.sv
// Module : tb_timing_4macro
//
// Cycle-accurate throughput benchmark for the 4-SRAM 256-bit bnn_top.
// Measures per-tile latency and full-frame inference time for conv2/conv3/conv4
// under back-to-back (no AXI stall) conditions — best-case chip throughput.
//
// Network dimensions (bnn_serengeti2.py):
//   conv2 : 112×112 output → 12,544 spatial, 64 out_ch → 802,816 tiles, 2 beats
//   conv3 :  56× 56 output →  3,136 spatial, 128 out_ch → 401,408 tiles, 3 beats
//   conv4 :  28× 28 output →    784 spatial, 256 out_ch → 200,704 tiles, 5 beats
//   Total : 1,404,928 tiles
//
// Throughput model per tile (2-phase SRAM read, 1 stall cycle per beat):
//   cycles ≈ 2×n_beats + pipeline_drain
//   Compare to 8-macro: cycles = n_beats + pipeline_drain (no stall)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_timing_4macro;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH  = 256;
    localparam int  WEIGHT_DEPTH  = 128;
    localparam int  MAX_BEATS     = 5;
    localparam real CLK_PERIOD_NS = 25.0;   // 40 MHz (P&R target)
    localparam real CLK_FREQ_MHZ  = 1000.0 / CLK_PERIOD_NS;

    // Network layer dimensions
    localparam int CONV2_SPATIAL = 12544;  localparam int CONV2_OUT_CH = 64;
    localparam int CONV3_SPATIAL =  3136;  localparam int CONV3_OUT_CH = 128;
    localparam int CONV4_SPATIAL =   784;  localparam int CONV4_OUT_CH = 256;
    localparam int CONV2_TILES   = CONV2_SPATIAL * CONV2_OUT_CH;  //  802,816
    localparam int CONV3_TILES   = CONV3_SPATIAL * CONV3_OUT_CH;  //  401,408
    localparam int CONV4_TILES   = CONV4_SPATIAL * CONV4_OUT_CH;  //  200,704
    localparam int CONV2_BEATS   = 2;
    localparam int CONV3_BEATS   = 3;
    localparam int CONV4_BEATS   = 5;

    // ── DUT ports ─────────────────────────────────────────────────────────────
    logic        clk, rst;
    logic        s_axis_tvalid, s_axis_tready;
    logic [VECTOR_WIDTH-1:0] s_axis_tdata;
    logic        s_axis_tlast;
    logic        m_axis_tvalid, m_axis_tready;
    logic [31:0] m_axis_tdata;
    logic        m_axis_tlast;
    logic        w_en;
    logic [6:0]  w_addr;   // 7-bit: WEIGHT_DEPTH=128 → $clog2(128)=7
    logic [31:0] w_data;
    logic [3:0]  cfg_beats_per_tile;
    logic [15:0] tile_count;

    bnn_top #(
        .VECTOR_WIDTH (VECTOR_WIDTH),
        .WEIGHT_DEPTH (WEIGHT_DEPTH),
        .MAX_BEATS    (MAX_BEATS)
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

    // ── Tasks ─────────────────────────────────────────────────────────────────

    task automatic do_reset();
        rst = 1; s_axis_tvalid = 0; s_axis_tdata = 0; s_axis_tlast = 0;
        m_axis_tready = 0; w_en = 0; w_addr = 0; w_data = 0;
        cfg_beats_per_tile = 5;
        repeat (4) @(posedge clk); #1;
        rst = 0;
        repeat (2) @(posedge clk); #1;
    endtask

    // Load all 128 logical words.
    // Each word: 8 consecutive w_en cycles at the same w_addr.
    // w_bank_sel[1:0] routes to bank (0-3), w_bank_sel[2] selects half.
    task automatic load_all_weights();
        for (int i = 0; i < WEIGHT_DEPTH; i++) begin
            for (int k = 0; k < 8; k++) begin
                @(posedge clk); #1;
                w_en   = 1'b1;
                w_addr = 7'(i);
                w_data = $urandom();
            end
        end
        @(posedge clk); #1;
        w_en = 1'b0;
    endtask

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

    task automatic drain_result();
        m_axis_tready = 1'b1;
        do @(posedge clk); while (!m_axis_tvalid);
        #1;
        m_axis_tready = 1'b0;
    endtask

    task automatic time_layer(
        input  int     n_tiles,
        input  int     n_beats,
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

    // ── Main ──────────────────────────────────────────────────────────────────
    initial begin

        // ── Single-tile latency ───────────────────────────────────────────────
        begin
            longint c2, c3, c4;

            $display("\n=== Single-tile latency (back-to-back, no stall) ===");

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV2_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV2_BEATS, c2);
            $display("  conv2 (2 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c2, c2*CLK_PERIOD_NS, CLK_FREQ_MHZ);

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV3_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV3_BEATS, c3);
            $display("  conv3 (3 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c3, c3*CLK_PERIOD_NS, CLK_FREQ_MHZ);

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV4_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(1, CONV4_BEATS, c4);
            $display("  conv4 (5 beats): %0d cycles  =  %.1f ns  @  %.0f MHz",
                     c4, c4*CLK_PERIOD_NS, CLK_FREQ_MHZ);
        end

        // ── Full-frame throughput ─────────────────────────────────────────────
        begin
            longint c2_total, c3_total, c4_total, frame_total;
            real    t2_ms, t3_ms, t4_ms, frame_ms;
            real    energy_mj;
            // Placeholder power — will be updated from post-PNR OpenSTA results
            localparam real POWER_MW = 12.007;  // post-route OpenSTA nom_tt_025C_1v80

            $display("\n=== Full-frame throughput (%0d + %0d + %0d = %0d total tiles) ===",
                     CONV2_TILES, CONV3_TILES, CONV4_TILES,
                     CONV2_TILES + CONV3_TILES + CONV4_TILES);

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV2_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV2_TILES, CONV2_BEATS, c2_total);
            t2_ms = (c2_total * CLK_PERIOD_NS) / 1_000_000.0;
            $display("  conv2: %0d tiles × %0d beats = %0d cycles  (%.1f ms)",
                     CONV2_TILES, CONV2_BEATS, c2_total, t2_ms);

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV3_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV3_TILES, CONV3_BEATS, c3_total);
            t3_ms = (c3_total * CLK_PERIOD_NS) / 1_000_000.0;
            $display("  conv3: %0d tiles × %0d beats = %0d cycles  (%.1f ms)",
                     CONV3_TILES, CONV3_BEATS, c3_total, t3_ms);

            do_reset(); load_all_weights(); repeat(2) @(posedge clk); #1;
            cfg_beats_per_tile = CONV4_BEATS;
            repeat(2) @(posedge clk); #1;
            time_layer(CONV4_TILES, CONV4_BEATS, c4_total);
            t4_ms = (c4_total * CLK_PERIOD_NS) / 1_000_000.0;
            $display("  conv4: %0d tiles × %0d beats = %0d cycles  (%.1f ms)",
                     CONV4_TILES, CONV4_BEATS, c4_total, t4_ms);

            frame_total = c2_total + c3_total + c4_total;
            frame_ms    = (frame_total * CLK_PERIOD_NS) / 1_000_000.0;
            energy_mj   = POWER_MW * frame_ms / 1000.0;

            $display("");
            $display("─────────────────────────────────────────────────────────");
            $display("  Total BNN cycles  : %0d", frame_total);
            $display("  Frame latency     : %.2f ms  (budget: 33.3 ms @ 30 FPS)", frame_ms);
            $display("  Throughput        : %.1f FPS (BNN layers only)", 1000.0 / frame_ms);
            $display("  Energy/frame      : %.3f mJ  (%.2f mW × %.2f ms)",
                     energy_mj, POWER_MW, frame_ms);
            $display("─────────────────────────────────────────────────────────");
        end

        $finish;
    end

endmodule

`default_nettype wire
