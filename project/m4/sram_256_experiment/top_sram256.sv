// =============================================================================
// File   : project/m4/sram_256_experiment/top_sram256.sv
// Module : bnn_top  (SRAM-backed weight memory, 32x256_8 macros)
//
// Uses sky130_sram_1kbyte_1rw1r_32x256_8 (479.8 × 397.5 µm) instead of
// the 32x512_8 macros. Smaller footprint → 4×4 grid instead of 8×1 row,
// so no single macro row spans the full die width.
//
// Memory architecture
// -------------------
// 4 columns × 4 rows = 16 macros total.
// Each column handles 32 bits of the 256-bit word (8 banks × 32 bits).
// Wait — simpler: 8 macros in parallel (one per 32-bit slice) × 2 rows deep
// (addr[8] selects upper/lower half of the 512-word address space).
//
// Actually: 8 parallel banks (256 bits wide) × 2 rows (512 entries total).
// Row select: addr[8] → upper half (macros 8-15) or lower half (macros 0-7).
// Each bank: 8-bit address → 256 entries per half → 512 total per bank.
//
// Physical layout: two rows of 8 macros, stacked vertically.
// Each row: 8 × 479.8 µm = 3,838 µm wide.
// Two rows: 2 × (397.5 + 50 µm spacing) = 895 µm tall.
// Dies at 5000 × 5000 µm leaves ~1,162 µm on each side for routing + compute.
//
// WEIGHT_DEPTH = 512 (same as 32x512_8 experiment, same RTL above this module).
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH  = 256,
    parameter int WEIGHT_DEPTH  = 512,
    parameter int MAX_BEATS     = 5
) (
    input  logic        clk,
    input  logic        rst,

    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic [VECTOR_WIDTH-1:0] s_axis_tdata,
    input  logic        s_axis_tlast,

    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tlast,

    input  logic        w_en,
    input  logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr,
    input  logic [VECTOR_WIDTH-1:0] w_data,

    input  logic [3:0]  cfg_beats_per_tile,
    output logic [15:0] tile_count
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    logic        core_valid, core_ready;
    logic [VECTOR_WIDTH-1:0] core_data;
    logic        frame_done;

    // =========================================================================
    // Weight memory — 16 × sky130_sram_1kbyte_1rw1r_32x256_8
    // 8 banks in parallel (256-bit word width) × 2 rows (512 entries deep).
    // addr[8] selects row; addr[7:0] is the within-row address.
    // =========================================================================
    localparam int W_ADDR_W = $clog2(WEIGHT_DEPTH);  // 9 bits for 512
    localparam int N_BANKS  = 8;
    localparam int N_ROWS   = 2;
    localparam int BANK_W   = 32;
    localparam int ROW_ADDR = 8;  // address bits per macro (256 entries)

    logic [W_ADDR_W-1:0]     w_ptr;
    logic [VECTOR_WIDTH-1:0] weight_word;

    logic [W_ADDR_W-1:0] sram_addr;
    assign sram_addr = w_en ? w_addr : w_ptr;

    // Row select and within-row address
    logic                row_sel;
    logic [ROW_ADDR-1:0] row_addr;
    assign row_sel  = sram_addr[W_ADDR_W-1];   // bit 8: selects upper/lower row
    assign row_addr = sram_addr[ROW_ADDR-1:0]; // bits 7:0: within-row address

    // Per-bank, per-row output wires
    logic [BANK_W-1:0] dout_row[N_ROWS][N_BANKS];

    genvar gb, gr;
    generate
        for (gr = 0; gr < N_ROWS; gr++) begin : g_sram_row
            for (gb = 0; gb < N_BANKS; gb++) begin : g_sram_bank
                sky130_sram_1kbyte_1rw1r_32x256_8 u_bank (
                    .clk0   (clk),
                    // Enable this row's macros only when row_sel matches.
                    // csb0 is active-low chip select.
                    .csb0   (row_sel != gr[0]),
                    .web0   (~w_en),
                    .wmask0 (4'b1111),
                    .addr0  (row_addr),
                    .din0   (w_data[gb*BANK_W +: BANK_W]),
                    .dout0  (dout_row[gr][gb]),
                    .clk1   (clk),
                    .csb1   (1'b1),
                    .addr1  ('0),
                    .dout1  ()
                );
            end
        end
    endgenerate

    // Mux read output from the selected row (registered — 1-cycle SRAM latency).
    // The row_sel registered here aligns with the SRAM's registered dout.
    logic row_sel_r;
    always_ff @(posedge clk)
        row_sel_r <= row_sel;

    genvar gw;
    generate
        for (gw = 0; gw < N_BANKS; gw++) begin : g_weight_mux
            assign weight_word[gw*BANK_W +: BANK_W] =
                row_sel_r ? dout_row[1][gw] : dout_row[0][gw];
        end
    endgenerate

    // =========================================================================
    // axis_interface
    // =========================================================================
    axis_interface u_axis_if (
        .aclk          (clk),
        .aresetn       (~rst),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tlast  (s_axis_tlast),
        .core_valid    (core_valid),
        .core_ready    (core_ready),
        .core_data     (core_data),
        .frame_done    (frame_done)
    );

    // =========================================================================
    // Weight address counter
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst)
            w_ptr <= '0;
        else if (core_valid && core_ready) begin
            if (w_ptr == W_ADDR_W'(WEIGHT_DEPTH - 1))
                w_ptr <= '0;
            else
                w_ptr <= w_ptr + 1'b1;
        end
    end

    // =========================================================================
    // Tile beat counter
    // =========================================================================
    localparam int BEAT_CTR_W = $clog2(MAX_BEATS + 1);
    logic [BEAT_CTR_W-1:0] beat_ctr;
    logic [3:0]             beats_this_tile;
    logic                   last_beat;

    assign last_beat = (beat_ctr == BEAT_CTR_W'(beats_this_tile - 1))
                        && core_valid && core_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            beat_ctr        <= '0;
            beats_this_tile <= 4'd5;
        end else if (core_valid && core_ready) begin
            if (last_beat) begin
                beat_ctr        <= '0;
                beats_this_tile <= cfg_beats_per_tile;
            end else begin
                beat_ctr <= beat_ctr + 1'b1;
            end
        end else if (beat_ctr == '0) begin
            beats_this_tile <= cfg_beats_per_tile;
        end
    end

    // =========================================================================
    // Pipeline delay (3 cycles = compute_core latency)
    // =========================================================================
    logic last_beat_r1, last_beat_r2, last_beat_r3;
    always_ff @(posedge clk) begin
        if (rst) begin
            last_beat_r1 <= 1'b0;
            last_beat_r2 <= 1'b0;
            last_beat_r3 <= 1'b0;
        end else begin
            last_beat_r1 <= last_beat;
            last_beat_r2 <= last_beat_r1;
            last_beat_r3 <= last_beat_r2;
        end
    end

    // =========================================================================
    // compute_core
    // =========================================================================
    logic signed [31:0] accum_out;
    logic               accum_clear;

    always_ff @(posedge clk)
        accum_clear <= rst ? 1'b0 : last_beat_r3;

    compute_core #(.VECTOR_WIDTH(VECTOR_WIDTH)) u_core (
        .clk        (clk),
        .rst        (rst),
        .s_valid    (core_valid),
        .s_ready    (core_ready),
        .accum_clear(accum_clear),
        .act_in     (core_data),
        .weight_in  (weight_word),
        .accum_out  (accum_out)
    );

    // =========================================================================
    // AXI4-Stream master output
    // =========================================================================
    logic [31:0] result_reg;
    logic        result_valid;

    always_ff @(posedge clk) begin
        if (rst) begin
            result_reg   <= '0;
            result_valid <= 1'b0;
        end else begin
            if (last_beat_r3 && (!result_valid || m_axis_tready)) begin
                result_reg   <= accum_out;
                result_valid <= 1'b1;
            end else if (result_valid && m_axis_tready) begin
                result_valid <= 1'b0;
            end
        end
    end

    assign m_axis_tvalid = result_valid;
    assign m_axis_tdata  = result_reg;
    assign m_axis_tlast  = result_valid;

    // =========================================================================
    // Tile counter
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst)
            tile_count <= '0;
        else if (last_beat_r3)
            tile_count <= tile_count + 1'b1;
    end

endmodule

`default_nettype wire
