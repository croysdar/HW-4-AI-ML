// =============================================================================
// File   : project/m4/sram_8macro_experiment/sim/tb_top_8macro.sv
// Module : tb_top_8macro
//
// Functional testbench for the 8-SRAM 256-bit bnn_top variant.
//
// Tests:
//   TEST 1: 1-beat tile, all-ones weights & activations.
//           XNOR(256'hFF..FF, 256'hFF..FF) = 256'hFF..FF → popcount=256
//           dot = 2*256−256 = 256.  Expected accumulator: 256.
//
//   TEST 2: 2-beat tile.
//           Beat 0: weights=all-1, acts=all-1 → dot=+256
//           Beat 1: weights=all-0, acts=all-0 → XNOR(0,0)=all-1 → dot=+256
//           Expected accumulator: 512.
//
//   TEST 3: 5-beat tile (MAX_BEATS), mixed pattern.
//           Weights row 0..4 = alternating 0xAAAA.../0x5555...
//           Acts all-ones.
//           XNOR(all-1, 0xAA..) = 0xAA.. → popcount=128 → dot=0
//           XNOR(all-1, 0x55..) = 0x55.. → popcount=128 → dot=0
//           Expected accumulator: 0 for all 5 beats.
//
//   TEST 4: Back-to-back tiles (pipelined throughput check).
//           Two 1-beat tiles issued back-to-back; results arrive in order.
//           Tile 0: acts=all-1, weights[0]=all-1 → 256
//           Tile 1: acts=all-0, weights[1]=all-0 → XNOR(0,0)=all-1 → 256
//
// Write protocol:
//   To load logical word W (0..255): assert w_en for 8 consecutive cycles,
//   hold w_addr=W, drive w_data = word[k*32+31 : k*32] for k=0..7.
//   The DUT's w_bank_sel counter routes each chunk to bank k automatically.
//   w_bank_sel resets to 0 on rst; must be aligned before each write group.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_top_8macro;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH = 256;
    localparam int  WEIGHT_DEPTH = 256;
    localparam int  MAX_BEATS    = 5;
    localparam real CLK_PERIOD   = 25.0;   // 40 MHz

    // ── DUT ports ─────────────────────────────────────────────────────────────
    logic        clk, rst;
    logic        s_axis_tvalid, s_axis_tready;
    logic [VECTOR_WIDTH-1:0] s_axis_tdata;
    logic        s_axis_tlast;
    logic        m_axis_tvalid, m_axis_tready;
    logic [31:0] m_axis_tdata;
    logic        m_axis_tlast;
    logic        w_en;
    logic [7:0]  w_addr;
    logic [31:0] w_data;
    logic [3:0]  cfg_beats_per_tile;
    logic [15:0] tile_count;

    // ── DUT ───────────────────────────────────────────────────────────────────
    bnn_top #(
        .VECTOR_WIDTH (VECTOR_WIDTH),
        .WEIGHT_DEPTH (WEIGHT_DEPTH),
        .MAX_BEATS    (MAX_BEATS)
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

    // ── Test tracking ─────────────────────────────────────────────────────────
    int pass_count, fail_count;

    task check(input string name, input int got, input int exp);
        if (got === exp) begin
            $display("PASS  %-40s  got=%0d", name, got);
            pass_count++;
        end else begin
            $display("FAIL  %-40s  got=%0d  expected=%0d", name, got, exp);
            fail_count++;
        end
    endtask

    // ── Reset ─────────────────────────────────────────────────────────────────
    task automatic do_reset();
        rst                <= 1'b1;
        s_axis_tvalid      <= 1'b0;
        s_axis_tdata       <= '0;
        s_axis_tlast       <= 1'b0;
        m_axis_tready      <= 1'b1;
        w_en               <= 1'b0;
        w_addr             <= 8'd0;
        w_data             <= 32'd0;
        cfg_beats_per_tile <= 4'd1;
        repeat (4) @(posedge clk); #1;
        rst <= 1'b0;
        repeat (2) @(posedge clk); #1;
    endtask

    // ── Write one 256-bit logical word to SRAM ────────────────────────────────
    // 8 consecutive w_en cycles, w_addr fixed.
    // Chunks: word[31:0] first (bank 0), word[255:224] last (bank 7).
    task automatic write_word(
        input int              logical_addr,
        input logic [255:0]    wval
    );
        for (int k = 0; k < 8; k++) begin
            @(posedge clk); #1;
            w_en   <= 1'b1;
            w_addr <= 8'(logical_addr);
            w_data <= wval[k*32 +: 32];
        end
        @(posedge clk); #1;
        w_en <= 1'b0;
    endtask

    // ── Send one AXI4-Stream beat ─────────────────────────────────────────────
    task automatic send_beat(
        input logic [VECTOR_WIDTH-1:0] data,
        input logic                    tlast
    );
        s_axis_tdata  <= data;
        s_axis_tlast  <= tlast;
        s_axis_tvalid <= 1'b1;
        do @(posedge clk); while (!s_axis_tready);
        #1;
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
    endtask

    // ── Wait for one tile result ──────────────────────────────────────────────
    task automatic wait_result(output logic signed [31:0] result);
        do @(posedge clk); while (!m_axis_tvalid);
        result = signed'(m_axis_tdata);
        #1;
    endtask

    // ── Common constants ──────────────────────────────────────────────────────
    localparam logic [255:0] ALL_ONES  = {256{1'b1}};
    localparam logic [255:0] ALL_ZEROS = {256{1'b0}};
    // Alternating nibbles: 1010...
    localparam logic [255:0] PATTERN_A = {32{8'hAA}};
    // Alternating nibbles: 0101...
    localparam logic [255:0] PATTERN_5 = {32{8'h55}};

    logic signed [31:0] result;

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        pass_count = 0;
        fail_count = 0;

        $display("=== 8-SRAM bnn_top functional testbench ===");

        // =====================================================================
        // TEST 1: 1-beat tile — all-ones weights and activations
        //   dot = 2·256 − 256 = 256
        // =====================================================================
        $display("\n--- TEST 1: 1-beat all-ones ---");
        do_reset();

        cfg_beats_per_tile <= 4'd1;
        write_word(0, ALL_ONES);
        repeat(2) @(posedge clk); #1;

        send_beat(ALL_ONES, 1'b1);
        wait_result(result);
        check("T1: dot(all-1, all-1)", int'(result), 256);

        // =====================================================================
        // TEST 2: 2-beat tile
        //   Beat 0: dot(all-1, all-1) = 256
        //   Beat 1: XNOR(all-0, all-0) = all-1 → dot = 256
        //   Accumulated: 512
        // =====================================================================
        $display("\n--- TEST 2: 2-beat tile ---");
        do_reset();

        cfg_beats_per_tile <= 4'd2;
        write_word(0, ALL_ONES);   // logical word 0 → weight_word for beat 0
        write_word(1, ALL_ZEROS);  // logical word 1 → weight_word for beat 1
        repeat(2) @(posedge clk); #1;

        send_beat(ALL_ONES,  1'b0);
        send_beat(ALL_ZEROS, 1'b1);
        wait_result(result);
        check("T2: accum(+256,+256)", int'(result), 512);

        // =====================================================================
        // TEST 3: 5-beat tile with alternating patterns
        //   Weights row 0,2,4 = 0xAA..  → XNOR(all-1, 0xAA..) = 0xAA.. → pop=128 → dot=0
        //   Weights row 1,3   = 0x55..  → XNOR(all-1, 0x55..) = 0x55.. → pop=128 → dot=0
        //   Accumulated: 0
        // =====================================================================
        $display("\n--- TEST 3: 5-beat zero-dot ---");
        do_reset();

        cfg_beats_per_tile <= 4'd5;
        write_word(0, PATTERN_A);
        write_word(1, PATTERN_5);
        write_word(2, PATTERN_A);
        write_word(3, PATTERN_5);
        write_word(4, PATTERN_A);
        repeat(2) @(posedge clk); #1;

        for (int b = 0; b < 5; b++)
            send_beat(ALL_ONES, (b == 4) ? 1'b1 : 1'b0);
        wait_result(result);
        check("T3: 5-beat accum=0", int'(result), 0);

        // =====================================================================
        // TEST 4: Back-to-back tiles (pipelined throughput)
        //   Tile 0: acts=all-1, weights[0]=all-1 → 256
        //   Tile 1: acts=all-0, weights[0]=all-0=XNOR→+256 — wait...
        //
        //   After tile 0 last_beat, r_ptr wraps: tile 1 beat 0 reads addr 0 again.
        //   But we already wrote word 0 = ALL_ONES and word 1 = ALL_ZEROS above.
        //   Reset clears r_ptr to 0, so tile 0 reads word 0 (all-1) and tile 1
        //   also reads word 0 (all-1) since r_ptr wraps back to 0.
        //   Tile 1: XNOR(all-0, all-1) = all-0 → popcount=0 → dot = −256.
        // =====================================================================
        $display("\n--- TEST 4: Back-to-back tiles ---");
        do_reset();

        cfg_beats_per_tile <= 4'd1;
        // Both words that tiles will read must be pre-loaded: SRAM retains state across tests.
        write_word(0, ALL_ONES);
        write_word(1, ALL_ZEROS);
        repeat(2) @(posedge clk); #1;

        // Tile 0: acts=all-1, weights[0]=all-1 → XNOR(1,1)=1 → pop=256 → dot=+256
        send_beat(ALL_ONES, 1'b1);
        wait_result(result);
        check("T4a: tile0 dot=+256", int'(result), 256);

        // Tile 1: acts=all-0, weights[1]=all-0 → XNOR(0,0)=1 → pop=256 → dot=+256
        send_beat(ALL_ZEROS, 1'b1);
        wait_result(result);
        check("T4b: tile1 dot=+256", int'(result), 256);

        // =====================================================================
        // Summary
        // =====================================================================
        $display("\n=== RESULTS: %0d PASS, %0d FAIL ===", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED");

        repeat (4) @(posedge clk);
        $finish;
    end

endmodule

`default_nettype wire
