// BNN convolutional compute core — top-level stub
// Implements XNOR-popcount dot product for 1-bit binary neural network layers.
// Corresponds to chiplet layers: BinarizeConv2d (conv2, conv3, conv4) in the
// 4-layer hybrid architecture.
//
// Interface: AXI4-Stream (256-bit, 300 MHz) — see project/m1/interface_selection.md
// Precision:  1-bit weights and activations; 32-bit signed popcount accumulator

module bnn_conv_core #(
    parameter int VECTOR_WIDTH = 256   // bits per input/weight word (matches AXI bus width)
) (
    input  logic                      clk,
    input  logic                      rst,          // active-high synchronous reset

    // AXI4-Stream input: packed binary activation vector
    input  logic [VECTOR_WIDTH-1:0]   s_axis_act,   // binary activations
    input  logic                      s_axis_valid,
    output logic                      s_axis_ready,

    // Packed binary weight vector (loaded separately; held stable during compute)
    input  logic [VECTOR_WIDTH-1:0]   weight,

    // Output: signed popcount accumulator
    output logic signed [31:0]        acc,          // running dot-product accumulator
    output logic                      acc_valid
);

    // XNOR: bit-wise equivalence of activation and weight
    logic [VECTOR_WIDTH-1:0] xnor_result;
    assign xnor_result = ~(s_axis_act ^ weight);

    // Popcount: count number of 1s in XNOR result (synthesized as adder tree by tools)
    logic [$clog2(VECTOR_WIDTH+1)-1:0] popcount;
    always_comb begin
        popcount = '0;
        for (int i = 0; i < VECTOR_WIDTH; i++)
            popcount += xnor_result[i];
    end

    // Convert popcount to signed dot-product contribution:
    //   dot = 2*popcount - VECTOR_WIDTH  (maps [0,N] → [-N, N])
    logic signed [31:0] dot;
    assign dot = 32'(signed'(popcount)) * 2 - 32'(VECTOR_WIDTH);

    // Accumulator register
    always_ff @(posedge clk) begin
        if (rst) begin
            acc       <= 32'sd0;
            acc_valid <= 1'b0;
        end else if (s_axis_valid && s_axis_ready) begin
            acc       <= acc + dot;
            acc_valid <= 1'b1;
        end else begin
            acc_valid <= 1'b0;
        end
    end

    assign s_axis_ready = 1'b1;  // always ready (no backpressure in this stub)

endmodule
