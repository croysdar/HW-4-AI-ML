# HW/SW Partition Rationale — Updated (Hybrid Architecture)

ECE 510 Spring 2026 — Hybrid Precision BNNClassifier (4-layer)

> **Version note:** This updates `partition_rationale.md` for the hybrid-precision architecture.
> The original (3-layer all-binary) is preserved in `partition_rationale.md`.

---

## 1. Bottleneck Identification

Profiling the CPU software baseline on M1 (`cProfile`, 100 passes) shows that `torch.conv2d` dominates runtime at 47.0% of total profiled time. The arithmetic intensity of the dominant kernel (conv1) is **12.34 FLOP/byte** (see `ai_calculation.md`), compared to the M1 ridge point of 38.1 FLOP/byte. The model is memory-bandwidth bound — the CPU fetches binary weights as 32-bit floats, consuming 32× more bandwidth than necessary.

With the updated 4-layer hybrid model, the whole-model AI rises to **57.9 FLOP/byte** (see `ai_calculation_new.md`), pushing past the M1 ridge. However, the dominant bottleneck remains the same: FP32 representation of 1-bit weights on a general-purpose CPU.

---

## 2. Partition Strategy

### Conv1 — Retained on Host ARM CPU (8-bit)

Conv1 is `nn.Conv2d(3→32, 3×3, stride=1)` operating on the full 224×224 RGB input. It is explicitly **not** binarized.

**Rationale:** Binarizing the raw 3-channel RGB input destroys spatial fidelity — the 1-bit approximation of low-contrast pixel values collapses to noise. Retaining Conv1 at 8-bit precision produces richer 32-channel feature maps that dramatically improve downstream binary layer accuracy. This accounts for the accuracy jump from 76.2% (all-binary 4-layer) to 85.2% (hybrid) with no additional parameters.

**Hardware implication:** Conv1 runs as a standard FP32 convolution on the host. Its output (32×224×224) is quantized to 8-bit and sent over AXI4-Stream to the chiplet (~1.53 MB per inference, ~46 MB/s at 30fps — well within the 8.0 GB/s effective AXI budget).

### Conv2–Conv4 — Moved to Chiplet XNOR Engine (1-bit)

All three `BinarizeConv2d` layers (32→64, 64→128, 128→256) execute on the chiplet using native 1-bit XNOR+Popcount logic.

**Rationale:** These layers collectively account for >90% of total MACs (1,386,633,024 of 1,473,970,176). At 1-bit precision on the chiplet, memory traffic for weights drops 32×, shifting the hardware AI to **379.1 FLOP/byte** — 10× past the M1 ridge and fully compute-bound. The chiplet keeps Conv1 output at 8-bit on input, then binarizes activations on-chip before each BNN layer. No mixed-precision MAC array is needed on the chiplet.

### BatchNorm, Pooling, Linear — Retained on Host

BatchNorm2d parameters are absorbed into the adjacent conv weights at inference time (standard BN folding). AdaptiveAvgPool2d and Linear(256→2) are negligible in compute and remain on the host ARM CPU.

---

## 3. Roofline Trajectory

| Stage | AI (FLOP/byte) | Platform | Status |
|---|---|---|---|
| Original 3-layer SW (conv1) | 12.34 | M1 CPU | Memory-bound |
| Original 3-layer SW (whole model) | 46.3 | M1 CPU | Near ridge |
| New 4-layer hybrid SW (whole model) | 57.9 | M5 CPU | Compute-bound |
| **Chiplet XNOR (Conv2-4, 1-bit)** | **379.1** | Chiplet | Deeply compute-bound |

The chiplet operating point at 379.1 FLOP/byte crosses both the M1 ridge (38.1) and the M5 ridge (~20 FLOP/byte). The workload is no longer memory-wall limited — the bottleneck shifts to the XNOR engine's throughput.

---

## 4. Interface Bandwidth Check

The chiplet receives 8-bit feature maps from Conv1 over AXI4-Stream.

| Item | Value |
|---|---|
| AXI data per inference | 32 × 224 × 224 × 1 B = 1.53 MB |
| At 30 fps | 1.53 MB × 30 = **45.9 MB/s** |
| AXI4-Stream rated bandwidth | 9.6 GB/s (256-bit @ 300 MHz) |
| Effective bandwidth (target) | 8.0 GB/s |
| Utilization at 30fps | 45.9 MB/s ÷ 8,000 MB/s = **0.57%** |

The 8-bit Conv1 output is 4× larger than the 1-bit equivalent would be (~1.53 MB vs ~0.38 MB), but at 30fps the interface is still only 0.57% utilized. There is ample headroom even if frame rate or batch size increases substantially.

---

## 5. Accuracy Impact of Partition

| Architecture | Val Accuracy | Night FAR |
|---|---|---|
| 3-layer all-binary SW baseline | 73.4% | ~93.7% |
| 4-layer all-binary | 76.2% | — |
| **Hybrid (Conv1 8-bit on host)** | **85.2%** | — |
| Hybrid + expanded data + Optuna | **87.1%** | 18.1% |

The hybrid partition is the single largest accuracy improvement, delivering +9% over the all-binary equivalent at no increase in chiplet complexity.
