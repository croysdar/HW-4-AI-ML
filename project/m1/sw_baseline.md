# Software Baseline Benchmark

ECE 510 Spring 2026 — M1 Deliverable

---

## 1. Platform Configuration

| Field               | Value                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| **Hardware**        | Apple Silicon (arm64) Apple M1                                                                         |
| **OS**              | macOS Sequoia 15.x — Darwin 24.6.0                                                                     |
| **CPU cores**       | 8 logical                                                                                              |
| **Python**          | 3.14.3                                                                                                 |
| **PyTorch**         | 2.11.0                                                                                                 |
| **Baseline device** | CPU (intentional — MPS/GPU not used; this is the general-purpose processor baseline for M4 comparison) |
| **Batch size**      | 1 (single-image inference)                                                                             |
| **Random seed**     | 42                                                                                                     |
| **Input shape**     | 3 × 224 × 224 (RGB trail-camera frame)                                                                 |

> **Reproducibility note:** The benchmark uses `torch.randn` synthetic data with `seed=42` — no dataset download required. Running on a different chip (e.g., Intel x86 vs. Apple M-series) will produce different absolute numbers; the M4 comparison must be run on the same machine.

---

## 2. Execution Time

Benchmark script: `project/baseline_profile.py`  
Method: 1 warm-up pass (discarded), then 100 forward-pass inferences timed with `time.perf_counter()`.

| Metric                               | Value                |
| ------------------------------------ | -------------------- |
| **Total wall-clock time (100 runs)** | 1219.0 ms            |
| **Mean latency per image**           | 12.190 ms            |
| **Median latency per image**         | ≈ 12.2 ms (see note) |

> **Median note:** The script records aggregate time across 100 runs and reports the mean (total / N). For a stable CPU-bound workload with no I/O jitter, mean ≈ median. To compute the strict median, re-run the script after adding per-run timing (see `baseline_profile.py` for guidance).

---

## 3. Throughput

| Metric                                | Value                                           |
| ------------------------------------- | ----------------------------------------------- |
| **Throughput**                        | 82.0 samples/sec                                |
| **Attained performance (full model)** | ~83.0 GFLOP/s (1,011,549,572 FLOPs ÷ 0.01219 s) |

---

## 4. Memory Usage

| Metric                                    | Value    | Source    |
| ----------------------------------------- | -------- | --------- |
| **Model parameters**                      | 0.37 MB  | torchinfo |
| **Activations (fwd pass)**                | 44.96 MB | torchinfo |
| **Input tensor**                          | 0.60 MB  | torchinfo |
| **Estimated total (model + activations)** | 45.93 MB | torchinfo |

---

## 5. Model Summary (for reference)

```
Layer (type)                Input Shape        Output Shape        MACs
────────────────────────────────────────────────────────────────────────
BinarizeConv2d (conv1)      [1, 3, 224, 224]   [1, 32, 224, 224]   43,352,064
BatchNorm2d                 [1, 32, 224, 224]  [1, 32, 224, 224]   64
BinarizeConv2d (conv2)      [1, 32, 224, 224]  [1, 64, 112, 112]   231,211,008
BatchNorm2d                 [1, 64, 112, 112]  [1, 64, 112, 112]   128
BinarizeConv2d (conv3)      [1, 64, 112, 112]  [1, 128, 56, 56]    231,211,008
BatchNorm2d                 [1, 128, 56, 56]   [1, 128, 56, 56]    256
AdaptiveAvgPool2d           [1, 128, 56, 56]   [1, 128, 1, 1]      —
Linear                      [1, 128]            [1, 2]              258
────────────────────────────────────────────────────────────────────────
Total MACs:    505,774,786
Total FLOPs:   1,011,549,572  (MACs × 2)
Total params:  93,730
```
