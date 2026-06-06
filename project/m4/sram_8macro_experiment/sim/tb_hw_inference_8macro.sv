// =============================================================================
// File   : project/m4/sram_8macro_experiment/sim/tb_hw_inference_8macro.sv
// Module : tb_hw_inference_8macro
//
// Hardware-inference co-simulation testbench for bnn_top (8-SRAM 256-bit variant).
//
// Driven by the same text-file protocol as the 1-macro variant, with two
// differences in the write protocol:
//
//   1-macro:  w_addr = logical_word * 8 + chunk   (chunk 0..7 at different rows)
//   8-macro:  w_addr = logical_word               (same addr for all 8 chunks;
//             DUT's internal w_bank_sel routes each chunk to the correct bank)
//
//   SRAM_LOGICAL_DEPTH = 256 (vs 32 for 1-macro — 8 banks × 32 bits = 256-bit wide)
//
// File formats (produced by run_hw_inference_8macro.py):
//
//   hw_inference_weights.txt
//     LOAD <256>
//     <256-bit hex word 0>  (64 hex chars, MSB-first)
//     ... (256 lines total, zero-padded)
//
//   hw_inference_stimulus.txt  — identical format to 1-macro version
//     # BATCH <n_tiles>
//     <cfg_beats> 0 <beat0_hex> [<beat1_hex> ...]
//
//   hw_inference_results.txt  — written by this testbench
//     One signed decimal integer per tile.
//
// TILES_PER_RELOAD (before w_ptr wraps):
//   n_beats=2 → floor(256/2) = 128
//   n_beats=3 → floor(256/3) = 85
//   n_beats=5 → floor(256/5) = 51
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_hw_inference_8macro;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH    = 256;
    localparam int  WEIGHT_DEPTH    = 256;
    localparam int  MAX_BEATS       = 5;
    localparam real CLK_PERIOD      = 3.333;  // ns — 300 MHz for fast co-sim

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

    initial clk = 1'b0;
    always  #(CLK_PERIOD / 2.0) clk = ~clk;

    // ── Variables ─────────────────────────────────────────────────────────────
    integer              wt_fd, stim_fd, res_fd, scan_ret;
    integer              n_words_in_load, n_tiles_batch;
    integer              cfg_beats_i, w_base_i;
    integer              total_tiles_done, b;

    logic [VECTOR_WIDTH-1:0] load_buf  [0:WEIGHT_DEPTH-1];
    logic [VECTOR_WIDTH-1:0] tile_beats[0:MAX_BEATS-1];
    logic [VECTOR_WIDTH-1:0] beat_hex  [0:MAX_BEATS-1];
    logic [VECTOR_WIDTH-1:0] word_hex;
    logic signed [31:0]      tile_result;
    string tag_str, line_buf;

    // ── Tasks ─────────────────────────────────────────────────────────────────

    task automatic do_reset();
        rst                <= 1'b1;
        s_axis_tvalid      <= 1'b0;
        s_axis_tdata       <= '0;
        s_axis_tlast       <= 1'b0;
        m_axis_tready      <= 1'b1;
        w_en               <= 1'b0;
        w_addr             <= 8'd0;
        w_data             <= 32'd0;
        cfg_beats_per_tile <= 4'd5;
        repeat (4) @(posedge clk); #1;
        rst <= 1'b0;
        repeat (2) @(posedge clk); #1;
    endtask

    // Write one 256-bit logical word using 8-macro protocol:
    // 8 consecutive w_en cycles, same w_addr, chunks 0..7 to banks 0..7.
    task automatic write_logical_word(
        input int           logical_addr,
        input logic [255:0] wval
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

    task automatic wait_result(output logic signed [31:0] result);
        do @(posedge clk); while (!m_axis_tvalid);
        result = signed'(m_axis_tdata);
        #1;
    endtask

    // ── Main ──────────────────────────────────────────────────────────────────
    initial begin
        wt_fd   = $fopen("hw_inference_weights.txt",  "r");
        stim_fd = $fopen("hw_inference_stimulus.txt", "r");
        res_fd  = $fopen("hw_inference_results.txt",  "w");

        if (wt_fd == 0) begin $display("ERROR: cannot open hw_inference_weights.txt");  $finish; end
        if (stim_fd == 0) begin $display("ERROR: cannot open hw_inference_stimulus.txt"); $finish; end
        if (res_fd == 0) begin $display("ERROR: cannot open hw_inference_results.txt");   $finish; end

        do_reset();
        total_tiles_done = 0;

        while (!$feof(wt_fd)) begin

            // ── Read LOAD header ──────────────────────────────────────────────
            scan_ret = 0;
            while (!$feof(wt_fd) && scan_ret < 2) begin
                if ($fgets(line_buf, wt_fd) == 0) break;
                scan_ret = $sscanf(line_buf, "%s %d", tag_str, n_words_in_load);
            end
            if ($feof(wt_fd) && scan_ret < 2) break;
            if (tag_str != "LOAD") begin
                $display("TB ERROR: expected LOAD, got '%s'", tag_str); $finish;
            end

            // ── Read n logical weight words ───────────────────────────────────
            for (int w = 0; w < n_words_in_load; w++) begin
                if ($fgets(line_buf, wt_fd) == 0) begin
                    $display("TB ERROR: premature EOF in weights at word %0d", w); $finish;
                end
                if ($sscanf(line_buf, "%h", word_hex) != 1) begin
                    $display("TB ERROR: failed to parse weight hex at word %0d", w); $finish;
                end
                load_buf[w] = word_hex;
            end

            // ── Reset DUT and write weights ───────────────────────────────────
            // Reset clears w_bank_sel to 0, aligning bank writes.
            do_reset();
            for (int w = 0; w < n_words_in_load; w++)
                write_logical_word(w, load_buf[w]);
            repeat (2) @(posedge clk); #1;

            // ── Read BATCH header ─────────────────────────────────────────────
            n_tiles_batch = 0;
            scan_ret = 0;
            while (!$feof(stim_fd) && scan_ret < 1) begin
                if ($fgets(line_buf, stim_fd) == 0) break;
                scan_ret = $sscanf(line_buf, "# BATCH %d", n_tiles_batch);
            end
            if (scan_ret < 1) begin
                $display("TB ERROR: expected '# BATCH <n>' in stimulus"); $finish;
            end

            // ── Run tiles ─────────────────────────────────────────────────────
            for (int t = 0; t < n_tiles_batch; t++) begin
                if ($fgets(line_buf, stim_fd) == 0) begin
                    $display("TB ERROR: premature EOF in stimulus at tile %0d", t); $finish;
                end
                scan_ret = $sscanf(line_buf, "%d %d %h %h %h %h %h",
                                   cfg_beats_i, w_base_i,
                                   beat_hex[0], beat_hex[1], beat_hex[2],
                                   beat_hex[3], beat_hex[4]);

                for (b = 0; b < cfg_beats_i; b++)
                    tile_beats[b] = beat_hex[b];

                cfg_beats_per_tile <= 4'(cfg_beats_i);
                @(posedge clk); #1;

                for (b = 0; b < cfg_beats_i; b++)
                    send_beat(tile_beats[b], (b == cfg_beats_i - 1) ? 1'b1 : 1'b0);

                wait_result(tile_result);
                $fdisplay(res_fd, "%0d", tile_result);
                total_tiles_done++;
            end

        end

        $fclose(wt_fd);
        $fclose(stim_fd);
        $fclose(res_fd);
        repeat (4) @(posedge clk);
        $display("HW_INFERENCE_DONE — %0d tiles processed", total_tiles_done);
        $finish;
    end

endmodule

`default_nettype wire
