`timescale 1ns/1ps
module mac_tb;
    logic              clk;
    logic              rst;
    logic signed [7:0] a;
    logic signed [7:0] b;
    logic signed [31:0] out;

    mac dut (.clk(clk), .rst(rst), .a(a), .b(b), .out(out));

    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; a = 0; b = 0;
        @(posedge clk); #1;

        // Deassert reset, apply a=3, b=4 for 3 cycles
        rst = 0; a = 3; b = 4;
        @(posedge clk); #1; $display("Cycle 1: out=%0d (expect 12)",  out);
        @(posedge clk); #1; $display("Cycle 2: out=%0d (expect 24)",  out);
        @(posedge clk); #1; $display("Cycle 3: out=%0d (expect 36)",  out);

        // Assert reset
        rst = 1; a = 0; b = 0;
        @(posedge clk); #1; $display("After rst: out=%0d (expect 0)", out);

        // Deassert reset, apply a=-5, b=2 for 2 cycles
        rst = 0; a = -5; b = 2;
        @(posedge clk); #1; $display("Cycle 5: out=%0d (expect -10)", out);
        @(posedge clk); #1; $display("Cycle 6: out=%0d (expect -20)", out);

        $finish;
    end
endmodule
