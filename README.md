# HW-4-AI-ML

Rebecca Gilbert-Croysdale

ECE 410/510 Spring 2026

## Project Topic:

Binary Neural Network

---

## HDL Compute Core (`project/hdl/`)

### Module: `bnn_conv_core.sv`

Implements the XNOR-popcount dot product for the chiplet's binary convolutional layers (conv2, conv3, conv4 in the 4-layer hybrid architecture). On each valid AXI4-Stream transaction, it computes the XNOR of a 256-bit activation word and a 256-bit weight word, counts the matching bits (popcount), converts to a signed dot-product contribution, and accumulates into a 32-bit signed register.

**Parameters:** `VECTOR_WIDTH` (default 256) — matches the AXI4-Stream bus width.
**Reset:** active-high synchronous.
**Precision:** 1-bit weights and activations; 32-bit signed accumulator.

### Interface Choice: AXI4-Stream (256-bit, 300 MHz)

From M1 analysis, the hardware chiplet targets 1200 GFLOP/s at an arithmetic intensity of 150 FLOP/byte, requiring a minimum memory bandwidth of 8.0 GB/s (see `project/m1/interface_selection.md`). AXI4-Stream at 256-bit / 300 MHz delivers a rated bandwidth of 9.6 GB/s — a 20% margin above the minimum — while its unidirectional, address-free design eliminates per-transaction overhead, keeping the chiplet in the compute-bound regime. SPI or I2C would be orders of magnitude too slow for streaming image tensors; AXI4-Lite adds unnecessary read-channel overhead for a streaming workload.

### Precision Choice: 1-bit weights and activations

Binary (1-bit) quantization is the defining feature of this project's BNN chiplet. XNOR-popcount replaces floating-point multiply-accumulate, reducing multiply cost to a single gate and addition cost to a bit-count. The M1 baseline showed the chiplet achieves 379.1 FLOP/byte arithmetic intensity — deeply compute-bound — confirming that precision reduction does not shift the bottleneck to memory bandwidth. The 32-bit signed accumulator preserves full popcount range (up to ±256 for a 256-bit vector width).
