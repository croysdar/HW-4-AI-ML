// Testbench for crossbar_mac
//
// Weights (row-major, weight[i][j]):
//   row 0: [ 1, -1,  1, -1]
//   row 1: [ 1,  1, -1, -1]
//   row 2: [-1,  1,  1, -1]
//   row 3: [-1, -1, -1,  1]
//
// Input: in = [10, 20, 30, 40]
//
// Hand-calculated expected outputs:
//   out[0] =  1*10 +  1*20 + (-1)*30 + (-1)*40 = 10+20-30-40 = -40
//   out[1] = (-1)*10 +  1*20 +  1*30 + (-1)*40 = -10+20+30-40 =   0
//   out[2] =  1*10 + (-1)*20 +  1*30 + (-1)*40 = 10-20+30-40 = -20
//   out[3] = (-1)*10 + (-1)*20 + (-1)*30 +  1*40 = -10-20-30+40 = -20
`timescale 1ns/1ps

module crossbar_tb;

    logic        clk, rst_n, weight_load;
    logic [3:0]  wrow0, wrow1, wrow2, wrow3;
    logic signed [7:0]  in0, in1, in2, in3;
    logic signed [15:0] out0, out1, out2, out3;

    crossbar_mac dut (
        .clk(clk), .rst_n(rst_n), .weight_load(weight_load),
        .wrow0(wrow0), .wrow1(wrow1), .wrow2(wrow2), .wrow3(wrow3),
        .in0(in0), .in1(in1), .in2(in2), .in3(in3),
        .out0(out0), .out1(out1), .out2(out2), .out3(out3)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    task automatic check(input string name, input signed [15:0] got, exp);
        if (got === exp)
            $display("  %s = %0d   PASS (expected %0d)", name, got, exp);
        else
            $display("  %s = %0d   FAIL (expected %0d)", name, got, exp);
    endtask

    initial begin
        $display("=== crossbar_mac testbench ===");
        $display("Weights: [[1,-1,1,-1],[1,1,-1,-1],[-1,1,1,-1],[-1,-1,-1,1]]");
        $display("Input:   [10, 20, 30, 40]");
        $display("Expected: out[0]=-40  out[1]=0  out[2]=-20  out[3]=-20");
        $display("");

        // Reset
        rst_n = 0; weight_load = 0;
        in0 = 0; in1 = 0; in2 = 0; in3 = 0;
        wrow0 = 4'b0; wrow1 = 4'b0; wrow2 = 4'b0; wrow3 = 4'b0;

        @(posedge clk); #1;
        rst_n = 1;

        // Load weights. Encoding: bit j of wrow_i = weight[i][j]; 1=+1, 0=-1.
        //   row 0: [+1,-1,+1,-1] → j=0:1, j=1:0, j=2:1, j=3:0 → 4'b0101
        //   row 1: [+1,+1,-1,-1] → j=0:1, j=1:1, j=2:0, j=3:0 → 4'b0011
        //   row 2: [-1,+1,+1,-1] → j=0:0, j=1:1, j=2:1, j=3:0 → 4'b0110
        //   row 3: [-1,-1,-1,+1] → j=0:0, j=1:0, j=2:0, j=3:1 → 4'b1000
        wrow0 = 4'b0101;
        wrow1 = 4'b0011;
        wrow2 = 4'b0110;
        wrow3 = 4'b1000;
        weight_load = 1;
        @(posedge clk); #1;
        weight_load = 0;

        // Apply inputs
        in0 = 8'sd10;
        in1 = 8'sd20;
        in2 = 8'sd30;
        in3 = 8'sd40;

        // Outputs are combinational — wait one delta after clock edge
        #1;

        $display("--- Simulation Results ---");
        check("out[0]", out0, -16'sd40);
        check("out[1]", out1,  16'sd0);
        check("out[2]", out2, -16'sd20);
        check("out[3]", out3, -16'sd20);

        $display("");
        $display("=== Simulation complete ===");
        $finish;
    end

endmodule
