// =============================================================================
// File   : project/m4/sram_experiment/top_sram.sv
// Module : bnn_top  (SRAM-backed weight memory variant)
//
// Description
// -----------
// SRAM-backed variant of bnn_top for the Option 5 experiment: integrate
// sky130_sram_2kbyte_1rw1r_32x512_8 OpenRAM macros for weight memory and use
// OpenLane 2's DRT_MIN_LAYER + GRT_LAYER_ADJUSTMENTS to force the router away
// from met1+met2 (the layers blocked by the macros' full-body obstructions).
//
// Memory architecture
// -------------------
// 8 macros in parallel for 256-bit word width:
//   bank[0]: bits[31:0]    bank[1]: bits[63:32]   bank[2]: bits[95:64]
//   bank[3]: bits[127:96]  bank[4]: bits[159:128] bank[5]: bits[191:160]
//   bank[6]: bits[223:192] bank[7]: bits[255:224]
// Each macro: 32-bit word × 512 entries → total 512 × 256-bit words.
//
// Port assignment per macro:
//   Port 0 (R+W, port 0 used for both load and read): driven by w_en/w_addr
//     during weight loading, and by w_ptr during compute.
//   Port 1 (R-only, unused in this design): tied off (csb1 = 1).
//
// SRAM has a 1-cycle read latency. We register w_ptr to align weight_word with
// the activation beat that read it. The compute_core pipeline is unchanged
// (3 stages); the SRAM read latency adds one extra stage to the weight path.
//
// Identical to register-file bnn_top in:
//   - AXI4-Stream slave/master interfaces
//   - axis_interface skid buffer
//   - compute_core (3-stage XNOR+popcount)
//   - tile FSM (beat counter, last_beat detection, accum_clear timing)
//   - Tile counter status output
//
// Differs only in the weight memory implementation.
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bnn_top #(
    parameter int VECTOR_WIDTH  = 256,
    parameter int WEIGHT_DEPTH  = 512,   // 8 macros × 512 entries (Port 0 only)
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
    // Weight memory — 8 × sky130_sram_2kbyte_1rw1r_32x512_8 macros
    // =========================================================================
    localparam int W_ADDR_W  = $clog2(WEIGHT_DEPTH);   // 9 bits for 512
    localparam int N_BANKS   = 8;                       // 8 × 32-bit = 256-bit word
    localparam int BANK_W    = 32;

    logic [W_ADDR_W-1:0]     w_ptr;
    logic [VECTOR_WIDTH-1:0] weight_word;

    // Mux: during weight load, address with w_addr; during compute, address with w_ptr.
    logic [W_ADDR_W-1:0] sram_addr;
    assign sram_addr = w_en ? w_addr : w_ptr;

    // 1-deep address pipeline to align registered SRAM output with compute pipeline.
    // SRAM has 1-cycle read latency; activations arrive at compute_core 1 cycle after
    // entering axis_interface skid buffer.
    logic        w_en_d;
    always_ff @(posedge clk) begin
        if (rst) w_en_d <= 1'b0;
        else     w_en_d <= w_en;
    end

    genvar gb;
    generate
        for (gb = 0; gb < N_BANKS; gb++) begin : g_sram_bank
            sky130_sram_2kbyte_1rw1r_32x512_8 u_bank (
                .clk0   (clk),
                .csb0   (1'b0),                              // chip select active-low: enabled
                .web0   (~w_en),                             // write enable active-low
                .wmask0 (4'b1111),                           // write all 4 bytes
                .addr0  (sram_addr),
                .din0   (w_data[gb*BANK_W +: BANK_W]),
                .dout0  (weight_word[gb*BANK_W +: BANK_W]),
                // Port 1 (R-only): unused, tied off
                .clk1   (clk),
                .csb1   (1'b1),                              // disabled
                .addr1  ('0),
                .dout1  ()                                    // unconnected
            );
        end
    endgenerate

    // =========================================================================
    // axis_interface (AXI4-Stream slave) — IDENTICAL to register-file design
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
    // Weight address counter — increments on every accepted activation beat.
    // The SRAM produces weight_word on the cycle AFTER w_ptr advances, which
    // aligns with the activation reaching compute_core (also +1 cycle delay
    // through axis_interface skid buffer).
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
    // Tile beat counter — IDENTICAL to register-file design
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
    // compute_core (3-stage pipelined) — IDENTICAL to register-file design
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
    // AXI4-Stream master output (1-deep register slice) — IDENTICAL
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
    // Tile counter — IDENTICAL
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst)
            tile_count <= '0;
        else if (last_beat_r3)
            tile_count <= tile_count + 1'b1;
    end

endmodule

`default_nettype wire
