# HW/SW Partition Rationale

ECE 510 Spring 2026 — Codefest 02 / M1 Deliverable

---

## 1. Bottleneck Identification

Profiling the CPU software baseline reveals that `torch.conv2d` is the dominant kernel, consuming **47.0% of total runtime**. This is entirely driven by the model's three `BinarizeConv2d` layers.

As calculated in `ai_calculation.md`, the arithmetic intensity (AI) of the `conv1` kernel is **12.34 FLOP/byte**. The Apple M1 ridge point is 38.1 FLOP/byte. Because the kernel’s AI sits 3.1× below this ridge point, the current implementation is strictly **memory-bandwidth bound**. The root inefficiency is data representation: the general-purpose CPU lacks a native 1-bit datapath, forcing it to fetch binary weights as 32-bit floats. This consumes 32× more memory bandwidth than necessary and leaves the compute units underutilized at just 83 GFLOP/s.

---

## 2. Partition Strategy

To resolve this bottleneck, we will partition the system as follows:

**Moved to Custom Hardware (Accelerator):**

- All three `BinarizeConv2d` layers.
- The hardware will utilize native 1-bit storage for weights/activations, replacing power-hungry FP32 MACs with highly efficient XNOR and Popcount spatial logic.

**Retained in Software (Host CPU):**

- Data loading, raw input binarization, `BatchNorm2d`, spatial pooling, and the final linear classifier.
- _Justification:_ These components collectively account for less than 8% of runtime. Offloading them would incur interface data-transfer overheads that far outweigh any potential compute latency reduction.

---

## 3. Roofline Trajectory

By utilizing native 1-bit storage in the accelerator, the data transfer volume per operation drops drastically. This shifts the kernel’s arithmetic intensity from 12.34 FLOP/byte to approximately **394.8 FLOP/byte**.

As shown in `roofline_project.png`, this shift safely crosses the 38.1 FLOP/byte ridge point, moving the workload into the **compute-bound region**. We target a conservative hardware design point of **150 FLOP/byte at 1,200 GFLOP/s**, eliminating the memory wall and maximizing throughput.

To sustain this 1,200 GFLOP/s target throughput at an arithmetic intensity of 150 FLOP/byte, the required memory bandwidth is **8.0 GB/s** (1200 GFLOP/s ÷ 150 FLOP/byte). We have chosen a 256-bit AXI4-Stream interface clocked at 300 MHz, providing a rated theoretical bandwidth of **9.6 GB/s**. Because the rated 9.6 GB/s exceeds the required 8.0 GB/s, the accelerator successfully avoids becoming interface-bound.
