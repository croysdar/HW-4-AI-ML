// =============================================================================
// File   : project/m4/sram_1macro_experiment/tb_hw_inference.sv
// Module : tb_hw_inference
//
// Hardware-inference co-simulation testbench for bnn_top (narrow-compute,
// single-SRAM-macro variant).
//
// Driven by two text files written by run_hw_inference.py:
//
//   hw_inference_weights.txt
//     One or more LOAD sections:
//       LOAD <n_logical_words>
//       <256-bit hex word 0>      ← 64 hex chars, MSB-first
//       ...
//     Each LOAD writes n_logical_words (= n_beats for one filter) logical words
//     at SRAM rows 0..(n_words*8-1), filling that many rows with the filter's
//     weight bits.  The same filter is reloaded periodically (every
//     TILES_PER_RELOAD tiles) to reset effective w_ptr alignment.
//
//   hw_inference_stimulus.txt
//     Corresponding BATCH sections (one per LOAD):
//       # BATCH <n_tiles>
//       <cfg_beats> 0 <beat0_hex> [<beat1_hex> ...]
//       ... (n_tiles lines total, w_base is always 0)
//
//   hw_inference_results.txt  (written by this testbench)
//     One signed decimal integer per tile (all tiles across all batches).
//
// Protocol:
//   The Python driver writes one (LOAD, BATCH) pair per "reload group".
//   A reload group consists of TILES_PER_RELOAD = floor(256/(n_beats*8))
//   consecutive spatial tiles that share the same filter weights.
//   After TILES_PER_RELOAD tiles, w_ptr has advanced exactly 256 SRAM rows
//   (= full wrap), so it returns to 0 for the next reload group.
//
//   TILES_PER_RELOAD:
//     n_beats=2 → 256/16 = 16 tiles per reload
//     n_beats=3 → 256/24 = 10 tiles per reload
//     n_beats=5 → 256/40 =  6 tiles per reload
//
// The testbench does NOT reset between tiles within a group (w_ptr auto-wraps).
// It resets the DUT once per LOAD (to clear the pipeline before writing weights).
// m_axis_tready is held high throughout.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_hw_inference;

    // ── Parameters ────────────────────────────────────────────────────────────
    localparam int  VECTOR_WIDTH    = 256;
    localparam int  CHUNK_WIDTH     = 32;
    localparam int  CHUNKS_PER_BEAT = 8;
    localparam int  WEIGHT_DEPTH    = 256;
    localparam int  LOGICAL_DEPTH   = WEIGHT_DEPTH / CHUNKS_PER_BEAT;  // 32
    localparam int  MAX_BEATS       = 5;
    localparam real CLK_PERIOD      = 3.333;  // ns — 300 MHz

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
    logic [7:0]  w_addr;
    logic [31:0] w_data;

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

    // ── Module-level variables ────────────────────────────────────────────────
    integer              wt_fd;
    integer              stim_fd;
    integer              res_fd;
    integer              scan_ret;
    integer              n_words_in_load;
    integer              n_tiles_batch;
    integer              cfg_beats_i;
    integer              w_base_i;
    integer              total_tiles_done;
    integer              b;

    logic [VECTOR_WIDTH-1:0] load_buf  [0:LOGICAL_DEPTH-1];
    logic [VECTOR_WIDTH-1:0] tile_beats[0:MAX_BEATS-1];
    logic [VECTOR_WIDTH-1:0] beat_hex  [0:MAX_BEATS-1];
    logic [VECTOR_WIDTH-1:0] word_hex;
    logic signed [31:0]      tile_result;

    string tag_str;
    string line_buf;

    // ── Tasks ─────────────────────────────────────────────────────────────────

    // Full DUT reset.  w_ptr → 0; pipeline cleared.  SRAM unaffected.
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

    // Write one 256-bit logical word to SRAM as 8 × 32-bit chunks.
    // Rows written: logical_addr*8 .. logical_addr*8+7.
    task automatic write_logical_word(
        input int           logical_addr,
        input logic [255:0] wval
    );
        integer base_row;
        base_row = logical_addr * CHUNKS_PER_BEAT;
        for (int c = 0; c < CHUNKS_PER_BEAT; c++) begin
            @(posedge clk); #1;
            w_en   <= 1'b1;
            w_addr <= 8'(base_row + c);
            w_data <= wval[c*CHUNK_WIDTH +: CHUNK_WIDTH];
        end
        @(posedge clk); #1;
        w_en <= 1'b0;
    endtask

    // Send one AXI4-Stream beat and wait for handshake.
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

    // Wait for a tile result (m_axis_tready high; result consumed immediately).
    task automatic wait_result(output logic signed [31:0] result);
        do @(posedge clk); while (!m_axis_tvalid);
        result = signed'(m_axis_tdata);
        #1;
    endtask

    // ── Main initial block ────────────────────────────────────────────────────
    initial begin
        wt_fd   = $fopen("hw_inference_weights.txt",  "r");
        stim_fd = $fopen("hw_inference_stimulus.txt", "r");
        res_fd  = $fopen("hw_inference_results.txt",  "w");

        if (wt_fd == 0) begin
            $display("ERROR: cannot open hw_inference_weights.txt");
            $finish;
        end
        if (stim_fd == 0) begin
            $display("ERROR: cannot open hw_inference_stimulus.txt");
            $finish;
        end
        if (res_fd == 0) begin
            $display("ERROR: cannot open hw_inference_results.txt for write");
            $finish;
        end

        do_reset();
        total_tiles_done = 0;

        // ── Main loop: one (LOAD, BATCH) pair per iteration ───────────────────
        // Each LOAD writes the same filter's weights; each BATCH processes the
        // corresponding group of spatial tiles using those weights.
        while (!$feof(wt_fd)) begin

            // ── 1. Read LOAD header ───────────────────────────────────────────
            scan_ret = 0;
            while (!$feof(wt_fd) && scan_ret < 2) begin
                if ($fgets(line_buf, wt_fd) == 0) break;
                scan_ret = $sscanf(line_buf, "%s %d", tag_str, n_words_in_load);
            end
            if ($feof(wt_fd) && scan_ret < 2) break;

            if (tag_str != "LOAD") begin
                $display("TB ERROR: expected LOAD, got '%s'", tag_str);
                $finish;
            end

            // ── 2. Read n logical weight words ────────────────────────────────
            for (int w = 0; w < n_words_in_load; w++) begin
                if ($fgets(line_buf, wt_fd) == 0) begin
                    $display("TB ERROR: premature EOF in weights at word %0d", w);
                    $finish;
                end
                if ($sscanf(line_buf, "%h", word_hex) != 1) begin
                    $display("TB ERROR: failed to parse weight hex at word %0d", w);
                    $finish;
                end
                load_buf[w] = word_hex;
            end

            // ── 3. Reset DUT, write weights to SRAM rows 0..n*8-1 ────────────
            // Reset clears w_ptr so it starts reading from row 0 for tile 0.
            do_reset();
            for (int w = 0; w < n_words_in_load; w++)
                write_logical_word(w, load_buf[w]);
            repeat (2) @(posedge clk); #1;

            // ── 4. Read BATCH header ──────────────────────────────────────────
            n_tiles_batch = 0;
            scan_ret = 0;
            while (!$feof(stim_fd) && scan_ret < 1) begin
                if ($fgets(line_buf, stim_fd) == 0) break;
                scan_ret = $sscanf(line_buf, "# BATCH %d", n_tiles_batch);
            end
            if (scan_ret < 1) begin
                $display("TB ERROR: expected '# BATCH <n>' in stimulus file");
                $finish;
            end

            // ── 5. Run each tile ──────────────────────────────────────────────
            // Tiles run back-to-back; w_ptr auto-increments.
            // Python ensures n_tiles_batch <= TILES_PER_RELOAD so w_ptr wraps
            // at most to the position it started, giving correct weight reads.
            for (int t = 0; t < n_tiles_batch; t++) begin
                if ($fgets(line_buf, stim_fd) == 0) begin
                    $display("TB ERROR: premature EOF in stimulus at tile %0d", t);
                    $finish;
                end
                // Format: "<cfg_beats> <w_base=0> <b0_hex> [<b1_hex> ...]"
                scan_ret = $sscanf(line_buf, "%d %d %h %h %h %h %h",
                                   cfg_beats_i, w_base_i,
                                   beat_hex[0], beat_hex[1], beat_hex[2],
                                   beat_hex[3], beat_hex[4]);

                for (b = 0; b < cfg_beats_i; b++)
                    tile_beats[b] = beat_hex[b];

                // Set cfg_beats_per_tile before first beat of this tile.
                cfg_beats_per_tile <= 4'(cfg_beats_i);
                @(posedge clk); #1;

                for (b = 0; b < cfg_beats_i; b++) begin
                    send_beat(tile_beats[b],
                              (b == cfg_beats_i - 1) ? 1'b1 : 1'b0);
                end

                wait_result(tile_result);
                $fdisplay(res_fd, "%0d", tile_result);
                total_tiles_done++;

            end  // for t

        end  // while !feof(wt_fd)

        $fclose(wt_fd);
        $fclose(stim_fd);
        $fclose(res_fd);

        repeat (4) @(posedge clk);
        $display("HW_INFERENCE_DONE — %0d tiles processed", total_tiles_done);
        $finish;
    end

endmodule

`default_nettype wire
