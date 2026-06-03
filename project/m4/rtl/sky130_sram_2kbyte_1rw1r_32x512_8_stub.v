// Lint/synthesis stub for sky130_sram_2kbyte_1rw1r_32x512_8.
// No timing constructs or $display — safe for Verilator and Yosys black-boxing.
// Yosys will use the liberty (.lib) for timing/area; this stub satisfies
// module resolution during elaboration without synthesizing any flip-flops.
// The real macro GDS/LEF is used during place-and-route.

module sky130_sram_2kbyte_1rw1r_32x512_8 (
    input  clk0,
    input  csb0,
    input  web0,
    input  [3:0] wmask0,
    input  [8:0] addr0,
    input  [31:0] din0,
    output [31:0] dout0,
    input  clk1,
    input  csb1,
    input  [8:0] addr1,
    output [31:0] dout1
);
    // Black box — implementation provided by PDK macro at P&R.
endmodule
