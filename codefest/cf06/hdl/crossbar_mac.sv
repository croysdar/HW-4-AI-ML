// 4x4 binary-weight crossbar MAC unit
// out[j] = sum_i weight[i][j] * in[i], weights are +1 or -1
// Weights stored in four 4-bit registers (one per row); bit j = weight[i][j]: 1=+1, 0=-1.
// Output is purely combinational from registered weights and current inputs.
module crossbar_mac (
    input  logic        clk,
    input  logic        rst_n,

    // Weight loading: hold weight_load high for one posedge
    input  logic        weight_load,
    input  logic [3:0]  wrow0,  // weights for input row 0: bit j = weight[0][j]
    input  logic [3:0]  wrow1,
    input  logic [3:0]  wrow2,
    input  logic [3:0]  wrow3,

    // 8-bit signed inputs
    input  logic signed [7:0] in0, in1, in2, in3,

    // 16-bit signed outputs (max magnitude = 4*128 = 512, fits in 10 bits; 16 gives margin)
    output logic signed [15:0] out0, out1, out2, out3
);

    logic [3:0] w0, w1, w2, w3;  // registered weight rows

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w0 <= 4'b0; w1 <= 4'b0; w2 <= 4'b0; w3 <= 4'b0;
        end else if (weight_load) begin
            w0 <= wrow0; w1 <= wrow1; w2 <= wrow2; w3 <= wrow3;
        end
    end

    // Sign-extend each input to 16 bits
    logic signed [15:0] t0, t1, t2, t3;
    assign t0 = {{8{in0[7]}}, in0};
    assign t1 = {{8{in1[7]}}, in1};
    assign t2 = {{8{in2[7]}}, in2};
    assign t3 = {{8{in3[7]}}, in3};

    // Combinational crossbar: for each output column j, sum ±inputs per weight bit j
    assign out0 = (w0[0] ? t0 : -t0) + (w1[0] ? t1 : -t1) + (w2[0] ? t2 : -t2) + (w3[0] ? t3 : -t3);
    assign out1 = (w0[1] ? t0 : -t0) + (w1[1] ? t1 : -t1) + (w2[1] ? t2 : -t2) + (w3[1] ? t3 : -t3);
    assign out2 = (w0[2] ? t0 : -t0) + (w1[2] ? t1 : -t1) + (w2[2] ? t2 : -t2) + (w3[2] ? t3 : -t3);
    assign out3 = (w0[3] ? t0 : -t0) + (w1[3] ? t1 : -t1) + (w2[3] ? t2 : -t2) + (w3[3] ? t3 : -t3);

endmodule
