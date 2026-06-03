// =============================================================================
// File   : project/m4/rtl/sram_behav_wrapper.sv
//
// Behavioral simulation wrapper for sky130_sram_2kbyte_1rw1r_32x512_8.
// Used ONLY for iverilog co-simulation (VERBOSE=0 avoids the $display/mem[]
// elaboration issue in Icarus). Synthesis uses the real PDK macro directly.
//
// Port names and timing match the OpenRAM model exactly:
//   Port 0 (RW): clk0, csb0 (active-low), web0 (active-low write), wmask0,
//                addr0[8:0], din0[31:0], dout0[31:0]
//   Port 1 (R):  clk1, csb1 (active-low), addr1[8:0], dout1[31:0]
// Both ports are synchronous (outputs registered on posedge clk).
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module sky130_sram_2kbyte_1rw1r_32x512_8 (
    input  logic        clk0,
    input  logic        csb0,
    input  logic        web0,
    input  logic [3:0]  wmask0,
    input  logic [8:0]  addr0,
    input  logic [31:0] din0,
    output logic [31:0] dout0,

    input  logic        clk1,
    input  logic        csb1,
    input  logic [8:0]  addr1,
    output logic [31:0] dout1
);

    logic [31:0] mem [0:511];

    // Port 0 — synchronous RW
    always_ff @(posedge clk0) begin
        if (!csb0) begin
            if (!web0) begin
                if (wmask0[0]) mem[addr0][ 7: 0] <= din0[ 7: 0];
                if (wmask0[1]) mem[addr0][15: 8] <= din0[15: 8];
                if (wmask0[2]) mem[addr0][23:16] <= din0[23:16];
                if (wmask0[3]) mem[addr0][31:24] <= din0[31:24];
            end
            dout0 <= mem[addr0];
        end
    end

    // Port 1 — synchronous R
    always_ff @(posedge clk1) begin
        if (!csb1)
            dout1 <= mem[addr1];
    end

endmodule

`default_nettype wire
