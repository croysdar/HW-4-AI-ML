// =============================================================================
// File   : project/m4/rtl/top_regfile.sv  (register-file variant — not the final design)
// Module : bnn_top
//
// Description
// -----------
// M4 integration top for the BNN XNOR-Popcount hardware accelerator chiplet.
// Synthesizes the full bnn_top datapath (AXI4-Stream interfaces + 3-stage
// pipelined compute_core + tile FSM) with a register-file weight memory
// at WEIGHT_DEPTH=640 words × 256 bits.
//
// Weight memory
// -------------
// WEIGHT_DEPTH=640 uses a 640×256-bit register file (~164K FFs). This holds
// the largest single sub-layer pass (128 conv4 output filters × 5 beats/filter
// = 640 words), enabling host-driven weight reload for layer/sub-layer tiling:
// a full inference performs 4 reload passes per frame (conv2, conv3, conv4
// upper-half, conv4 lower-half). At 30 FPS this is ~280 KB/s AXI traffic —
// negligible. The full production WEIGHT_DEPTH=1792 (256 KB) cannot be held
// on-chip with a register file due to Yosys/ABC FF count limits and would
// require SRAM macros. SRAM macro integration was attempted but the available
// sky130_sram_*_1rw1r macros carry full-body met1+met2 obstructions that
// cause routing congestion (GRT-0118) regardless of die size or macro
// arrangement. This is documented in the M4 README and synthesis_notes.md.
//
// Port Descriptions
// -----------------
// clk                 in   1      System clock (300 MHz target)
// rst                 in   1      Active-high synchronous reset
// s_axis_*            in/out      AXI4-Stream slave (activation input)
// m_axis_*            out/in      AXI4-Stream master (result output)
// w_en                in   1      Write enable: load one weight word
// w_addr              in   10     Word address (0..639)
// w_data              in   256    Weight word (256-bit)
// cfg_beats_per_tile  in   4      Beats per filter tile (2/3/5)
// tile_count          out  16     Completed tile count since reset
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH  = 256,
    parameter int WEIGHT_DEPTH  = 640,
    parameter int MAX_BEATS     = 5
) (
    input  logic        clk,
    input  logic        rst,

    // AXI4-Stream Slave — activation input
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic [VECTOR_WIDTH-1:0] s_axis_tdata,
    input  logic        s_axis_tlast,

    // AXI4-Stream Master — result output
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tlast,

    // Weight sideband load port
    input  logic        w_en,
    input  logic [$clog2(WEIGHT_DEPTH)-1:0] w_addr,
    input  logic [VECTOR_WIDTH-1:0] w_data,

    // Runtime tile configuration
    input  logic [3:0]  cfg_beats_per_tile,

    // Status
    output logic [15:0] tile_count
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    logic        core_valid;
    logic        core_ready;
    logic [VECTOR_WIDTH-1:0] core_data;
    logic        frame_done;

    // =========================================================================
    // Weight memory — 64×256-bit register file
    // =========================================================================
    localparam int W_ADDR_W = $clog2(WEIGHT_DEPTH);

    logic [VECTOR_WIDTH-1:0] weight_mem [0:WEIGHT_DEPTH-1];
    logic [VECTOR_WIDTH-1:0] weight_word;
    logic [W_ADDR_W-1:0]     w_ptr;

    always_ff @(posedge clk)
        if (w_en) weight_mem[w_addr] <= w_data;

    assign weight_word = weight_mem[w_ptr];

    // =========================================================================
    // axis_interface (AXI4-Stream slave)
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
    // Pipeline delay shift register (3 cycles = compute_core latency)
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
    // compute_core (3-stage pipelined)
    // =========================================================================
    logic signed [31:0] accum_out;
    logic               accum_clear;

    always_ff @(posedge clk)
        accum_clear <= rst ? 1'b0 : last_beat_r3;

    compute_core #(
        .VECTOR_WIDTH(VECTOR_WIDTH)
    ) u_core (
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
    // AXI4-Stream master output (1-deep register slice)
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
