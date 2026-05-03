# M2 Precision Strategy: Heterogeneous INT8 / 1-bit Hybrid

**Project:** Ultra-Low-Power BNN Trail Cam Smart Filter
**Course:** ECE 510 — Hardware for AI, Spring 2026
**Author:** Rebecca Gilbert-Croysdale

---

## 1. Numeric Formats by Layer

| Layer                     | Format                   | Location       | Justification                                                      |
| ------------------------- | ------------------------ | -------------- | ------------------------------------------------------------------ |
| Conv1 (3→32, 3×3)         | **INT8 fixed-point**     | ARM Host CPU   | Preserves high-fidelity spatial features at the RGB input boundary |
| Conv2 (32→64, 3×3, s=2)   | **1-bit (XNOR)**         | Custom chiplet | XNOR+Popcount engine; bitwise ops replace multiply-accumulate      |
| Conv3 (64→128, 3×3, s=2)  | **1-bit (XNOR)**         | Custom chiplet | Same engine; doubles channel depth                                 |
| Conv4 (128→256, 3×3, s=2) | **1-bit (XNOR)**         | Custom chiplet | Widens to 256 channels for discriminative power                    |
| BatchNorm + Pool + FC     | FP32 (host post-process) | ARM Host CPU   | Low compute, high dynamic range needed for final logits            |

Conv1 is retained as an **INT8 fixed-point** layer on the ARM host CPU rather than as 1-bit, because binarizing the first layer — which operates directly on raw 8-bit RGB pixel values — produces an unacceptable accuracy collapse. Retaining Conv1 at INT8 (rather than full FP32) simulates the precision available on a low-power MCU with a hardware INT8 DSP unit, without requiring a full FP32 multiply pipeline.

---

## 2. Roofline Analysis: Why Quantization Defeats the Memory Wall

The M1 roofline analysis established that the unquantized 1-bit chiplet path targets **1200 GFLOP/s** at an arithmetic intensity of **150 FLOP/byte**, requiring a minimum memory bandwidth of 8.0 GB/s. Our 256-bit AXI4-Stream interface at 300 MHz delivers **9.6 GB/s** rated bandwidth — a 20% margin — keeping the chiplet firmly in the compute-bound regime.

The decision to quantize Conv1 to INT8 extends this same logic to the host-CPU partition. A full FP32 activation tensor at the Conv1 output has width 32 channels × 224 × 224 pixels × 4 bytes/value = **6.4 MB** per image. Compressing that representation to INT8 (1 byte/value) shrinks the payload to **1.6 MB** — a **4× reduction** in the data that must cross the AXI4-Stream interface from host to chiplet. This directly reduces the effective arithmetic intensity demand on the bus, relaxing the bandwidth constraint and reinforcing the compute-bound operating point identified in M1.

Without INT8 compression, the feature-map transfer alone risks pushing the system toward the memory-bound region of the roofline, negating the efficiency gains from the XNOR+Popcount engine. With INT8, the data payload shrinks to fit comfortably within the 9.6 GB/s rated interface bandwidth, allowing the chiplet's 1-bit core to remain the binding compute resource — not the interconnect.

---

## 3. Validation Experiment: 100-Sample Comparison

A PyTorch fake-quantization script (`project/m2/validate_precision.py`) was used to compare the **FP32 Reference** model against the **INT8/1-bit Hybrid DUT** on exactly 100 held-out test images (50 blank, 50 non-blank), drawn from `data_20k/test/`.

The DUT uses symmetric per-tensor INT8 fake quantization on Conv1 weights and activations: scale = max(|x|) / 127, quantize to the nearest integer in [−128, 127], then dequantize by multiplying by scale. This faithfully simulates the rounding error introduced by an INT8 fixed-point unit.

| Metric                               | Value      |
| ------------------------------------ | ---------- |
| Reference (FP32 Conv1) accuracy      | 63.0%      |
| DUT (INT8/1-bit) accuracy            | 59.0%      |
| **Accuracy Delta (DUT − Reference)** | **−4.0%**  |
| **Logit MAE**                        | **0.2453** |
| **Logit Max Error**                  | **0.8360** |

The logit MAE of **0.2453** and max error of **0.8360** represent the pre-softmax output perturbation introduced by INT8 rounding on Conv1. These errors are small relative to the typical logit spread (which spans several units at high-confidence predictions), confirming that the quantization noise is well within the classification margin for the majority of samples.

Note on 1-bit precision: The 1-bit precision of Conv2, Conv3, and Conv4 is native to the network architecture via Quantization-Aware Training (QAT) in the baseline checkpoint. Both the PyTorch Reference model and the DUT evaluate these layers identically. Therefore, the quantization error and accuracy delta reported above strictly isolate the impact of applying Post-Training Quantization (PTQ) to the INT8 Conv1 layer.

---

## 4. Statement of Acceptability

The −4.0% accuracy delta from INT8 quantization on Conv1 is **acceptable** for this deployment scenario. The primary mission of the Serengeti2 wildlife camera system is **high recall for rare and endangered species** operating on a severely constrained **solar power budget** at a remote edge node.

In this context, recall is the dominant performance metric — a missed animal is a lost conservation record that cannot be recovered. The system already uses a **temporal 3-frame trigger** (Section 3.1, architectural roadmap) which suppresses false alarms exponentially (FAR drops from 13.1% to 0.22% with N=3 frames) without touching recall. The −4.0% accuracy degradation from INT8 primarily affects borderline-confidence frames where softmax probabilities are close to the 0.5 threshold; high-confidence detections are nearly unaffected, as confirmed by the small logit MAE relative to the logit range.

The power savings from running Conv1 at INT8 rather than FP32 on the host ARM CPU are non-trivial: INT8 MACs consume roughly 4× less energy than FP32 on ARM Cortex-class processors (per published Arm Cortex-M55 energy benchmarks). On a solar-harvested edge node where the daily energy budget is measured in millijoules, this reduction is mission-critical. Sacrificing 4 percentage points of accuracy on a 100-image probe to achieve this power reduction is a sound engineering trade-off — particularly given that the full-dataset accuracy of 87.6% (87.1% without distillation) and 95.3% night recall demonstrate that the model retains strong detection capability across the deployment distribution.
