# Software Baseline Benchmark — Updated (Hybrid Architecture)

ECE 510 Spring 2026

> **Version note:** This updates `sw_baseline.md` for the hybrid-precision 4-layer architecture running on M5.
> The original (3-layer all-binary, M1) is preserved in `sw_baseline.md`.

---

## 1. Platform Configuration

| Field | Value |
|---|---|
| **Hardware** | Apple Silicon (arm64) Apple M5 |
| **OS** | macOS (Darwin 25.4.0) |
| **Python** | 3.14.3 |
| **PyTorch** | 2.11.0 |
| **Baseline device** | CPU (intentional — MPS/GPU not used; general-purpose processor baseline) |
| **Batch size** | 1 (single-image inference) |
| **Random seed** | 42 |
| **Input shape** | 3 × 224 × 224 (RGB trail-camera frame) |

---

## 2. Architecture — 4-Layer Hybrid Precision

```
Layer                              Type                 Params    Partition
────────────────────────────────────────────────────────────────────────────
Conv2d (3→32, 3×3, s=1, p=1)      nn.Conv2d (8-bit)       864    Host CPU
BatchNorm2d(32)                    BN                       64    Host CPU
BinarizeConv2d (32→64, s=2, p=1)  1-bit BNN            18,432    Chiplet
BatchNorm2d(64)                    BN                      128    Host CPU
BinarizeConv2d (64→128, s=2, p=1) 1-bit BNN            73,728    Chiplet
BatchNorm2d(128)                   BN                      256    Host CPU
BinarizeConv2d (128→256, s=2,p=1) 1-bit BNN           294,912    Chiplet
BatchNorm2d(256)                   BN                      512    Host CPU
AdaptiveAvgPool2d(1×1)             Pool                      0    Host CPU
Linear(256→2)                      FC                      514    Host CPU
────────────────────────────────────────────────────────────────────────────
Total params:  389,410
```

Key change from original: Conv1 is now `nn.Conv2d` (standard 8-bit) rather than `BinarizeConv2d`. Conv4 (128→256) is new. See `partition_rationale_new.md` for full justification.

---

## 3. Execution Time

Method: 1 warm-up pass (discarded), then 100 forward-pass inferences, CPU device only.

| Metric | M1 (original 3-layer) | M5 (new 4-layer hybrid) |
|---|---|---|
| Total wall-clock (100 runs) | 1,219.0 ms | 1,267.4 ms |
| Mean latency per image | 12.190 ms | 12.674 ms |
| Throughput | 82.0 img/s | 78.9 img/s |

The new model is ~4% slower on CPU due to the larger 4-layer architecture (389k params vs 93k). In hardware, Conv2-4 move to the chiplet, so this SW baseline is intentionally pessimistic.

---

## 4. Throughput

| Metric | M1 (3-layer) | M5 (4-layer hybrid) |
|---|---|---|
| Throughput | 82.0 samples/sec | 78.9 samples/sec |
| Total FLOPs | 1,011,549,572 | 1,473,970,176 |
| Attained performance | 83.0 GFLOP/s | **116.3 GFLOP/s** |

The M5 delivers 40% more attained GFLOP/s despite running a larger model, reflecting the improved CPU compute throughput.

---

## 5. Memory Usage

| Metric | Value |
|---|---|
| Model parameters | 1.49 MB (389,410 × 4B FP32) |
| Checkpoint size (trained) | 4.70 MB (includes optimizer state) |
| Activations (fwd pass, approx) | ~54 MB (larger than original due to conv4) |

---

## 6. Arithmetic Intensity

Full derivation in `codefest/cf02/analysis/ai_calculation_new.md`.

| Layer | AI (FLOP/byte) |
|---|---|
| conv1 (host, 8-bit SW) | 12.3 |
| conv2 (chiplet, 1-bit HW) | 47.6 SW → 379.1 HW |
| conv3 | 90.5 SW → (included in 379.1 HW) |
| conv4 | 128.9 SW → (included in 379.1 HW) |
| **Whole model (SW float32)** | **57.9** |
| **Hardware chiplet (1-bit)** | **379.1** |

The M1 ridge point is 38.1 FLOP/byte. The M5 ridge is ~25 FLOP/byte (estimated). Both CPU SW baselines operate at or above the ridge in the compute-bound regime. The chiplet at 379.1 is deeply compute-bound on both platforms.
