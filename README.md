# HW-4-AI-ML

Rebecca Gilbert-Croysdale

ECE 410/510 Spring 2026

## Project: Ultra-Low-Power BNN Trail Cam Smart Filter

---

## HDL Compute Core (`project/hdl/`)

### Module: `bnn_conv_core.sv`

Implements the XNOR-popcount dot product for the chiplet's binary convolutional layers (conv2, conv3, conv4) in the 4-layer hybrid architecture. Conv1 (3→32, 3×3) runs as an INT8 fixed-point layer on the ARM host CPU; only conv2–4 execute on the chiplet. On each valid AXI4-Stream transaction, the core computes the XNOR of a 256-bit activation word and a 256-bit weight word, counts the matching bits (popcount), converts to a signed dot-product contribution, and accumulates into a 32-bit signed register.

**Parameters:** `VECTOR_WIDTH` (default 256) — matches the AXI4-Stream bus width.
**Reset:** active-high synchronous.
**Precision:** 1-bit weights and activations (Conv2–4 on chiplet); Conv1 runs INT8 on ARM host CPU. 32-bit signed accumulator.

### Interface Choice: AXI4-Stream (256-bit, 300 MHz)

**1. Peak Roofline Capability (M1 Constraint)**
From our M1 roofline analysis, the hardware chiplet targets a peak theoretical throughput of 1200 GFLOP/s. At an arithmetic intensity of 150 FLOP/byte, sustaining this absolute maximum compute rate requires an interface bandwidth of at least 8.0 GB/s (see `project/m1/interface_selection.md`). AXI4-Stream configured at 256-bit and 300 MHz delivers a rated bandwidth of 9.6 GB/s, providing a 20% safety margin. This ensures that the interface will never bottleneck the chiplet when it operates at maximum burst capacity. Furthermore, AXI4-Stream's unidirectional, address-free design eliminates per-transaction overhead, keeping the system firmly in the compute-bound regime. Slower protocols like SPI or I2C would be orders of magnitude too slow for streaming dense image tensors, while AXI4-Lite would introduce unnecessary read-channel latency for what is purely a streaming workload.

**2. Actual 30 FPS Deployment Bandwidth**
In actual deployment, the system processes video at 30 FPS. Because Conv1 runs on the ARM host CPU in INT8, the feature map payload sent to the chiplet per frame is exactly 1.6 MB (32 channels × 224 × 224 bytes). At 30 FPS, the actual continuous bandwidth requirement is only ~48 MB/s. Furthermore, because we use a Weight-Stationary dataflow for Conv2–4, the tiny 1-bit weights (~47 KB) are only loaded across the bus once during a Day/Night context switch, consuming 0 MB/s during active inference.

**3. Justification for AXI4-Stream**
While a ~50 MB/s continuous stream could theoretically be handled by a simpler bus (like parallel SPI or SDIO), AXI4-Stream was specifically chosen for this architecture because:

- **Native Integration:** It is the industry-standard streaming protocol for ARM SoCs to communicate with custom accelerators, requiring zero complex glue logic.
- **Zero Overhead:** As a unidirectional, address-free protocol, it eliminates the read/write address channel overhead inherent in AXI4-Lite or PCIe, perfectly matching the streaming nature of continuous image tensors.
- **Race-to-Sleep (Duty Cycling):** Because the BNN acts as a continuous 24/7 wake-up sensor at 30 FPS, the ARM host CPU must minimize its active time to save battery. The massive 9.6 GB/s headroom allows the CPU to blast a 1.6 MB frame to the chiplet in a fraction of a millisecond and immediately return to a deep sleep state. A slower, narrower bus would force the host CPU to stay awake constantly to stream the data, destroying the system's power budget.

The decision to run Conv1 at INT8 (rather than FP32) further relaxes this bandwidth constraint. The Conv1 output feature map (32 channels × 224 × 224) is 6.4 MB at FP32 but only 1.6 MB at INT8 — a 4× reduction in the payload that must cross the AXI4-Stream interface from host to chiplet. This keeps the effective bus demand well within the 9.6 GB/s rated bandwidth and reinforces the compute-bound operating point for the XNOR+Popcount engine (see `project/m2/precision.md`).

### Precision Choice: Hybrid INT8 (Conv1, host) + 1-bit XNOR (Conv2–4, chiplet)

The architecture uses heterogeneous precision. Conv1 is retained at INT8 fixed-point on the ARM host CPU: binarizing the layer that operates directly on raw 8-bit RGB pixels produces unacceptable accuracy collapse, and an INT8 MAC unit would require 4–8× the transistor area of a 1-bit XNOR gate on-chip while sitting idle during the three binary layers that dominate inference time. Fake-quantization experiments confirm a −4.0% accuracy delta and logit MAE of 0.2453 for INT8 vs. FP32 Conv1 — acceptable for this high-recall, power-constrained deployment (see `project/m2/precision.md`).

Conv2–4 run as 1-bit XNOR-popcount on the chiplet. Binary quantization reduces multiply cost to a single gate and addition cost to a bit-count. The M1 baseline showed the chiplet achieves 379.1 FLOP/byte arithmetic intensity — deeply compute-bound — confirming that 1-bit precision does not shift the bottleneck to memory bandwidth. The 32-bit signed accumulator preserves the full popcount range (up to ±256 per 256-bit vector) with ample headroom for multi-vector accumulation across kernel windows.

## 6. Reproducibility

**Simulator:** Icarus Verilog (iverilog) v13.0

To compile and run the `compute_core` self-checking testbench, execute the following commands from the repository root:

```bash
mkdir -p project/m2/sim

# Compute core (XNOR-Popcount engine)
iverilog -g2012 -o project/m2/sim/compute_core.vvp \
  project/m2/rtl/compute_core.sv project/m2/tb/tb_compute_core.sv
vvp project/m2/sim/compute_core.vvp | tee project/m2/sim/compute_core_run.log

# AXI4-Stream interface (skid buffer)
iverilog -g2012 -o project/m2/sim/interface.vvp \
  project/m2/rtl/interface.sv project/m2/tb/tb_interface.sv
vvp project/m2/sim/interface.vvp | tee project/m2/sim/interface_run.log
```

Both logs end with `VERIFIABLE PASS`. See `project/m2/README.md` for full details and waveform instructions.
