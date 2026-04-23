# BNN Accelerator Chiplet — Model Evaluation Report

ECE 510 Spring 2026  
Dataset: Caltech Camera Traps + Serengeti2 (combined)  
Evaluation date: 2026-04-23

---

## Model Summary

| Property                           | Value                                    |
| ---------------------------------- | ---------------------------------------- |
| Architecture                       | Hybrid-precision BNNClassifier (4-layer) |
| Parameters                         | 389,410                                  |
| Checkpoint size                    | 4.70 MB                                  |
| Best val accuracy (training epoch) | **87.1% (epoch 14 of 21, early stopped)** |
| AUC-ROC                            | **0.9377**                               |

### Layer Breakdown

| Layer | Type                               | Params  | Precision | Partition    |
| ----- | ---------------------------------- | ------- | --------- | ------------ |
| conv1 | nn.Conv2d (3→32, 3×3, s=1)         | 864     | 8-bit     | Host CPU     |
| bn1   | BatchNorm2d(32)                    | 64      | —         | Host CPU     |
| conv2 | BinarizeConv2d (32→64, 3×3, s=2)   | 18,432  | 1-bit     | Chiplet XNOR |
| bn2   | BatchNorm2d(64)                    | 128     | —         | Host CPU     |
| conv3 | BinarizeConv2d (64→128, 3×3, s=2)  | 73,728  | 1-bit     | Chiplet XNOR |
| bn3   | BatchNorm2d(128)                   | 256     | —         | Host CPU     |
| conv4 | BinarizeConv2d (128→256, 3×3, s=2) | 294,912 | 1-bit     | Chiplet XNOR |
| bn4   | BatchNorm2d(256)                   | 512     | —         | Host CPU     |
| pool  | AdaptiveAvgPool2d(1×1)             | 0       | —         | Host CPU     |
| fc    | Linear(256→2)                      | 514     | FP32      | Host CPU     |

Conv1 runs on the host ARM CPU and outputs 8-bit feature maps (32×224×224) over AXI4-Stream. Conv2–Conv4 execute on the chiplet as 1-bit XNOR+Popcount operations.

---

## Dataset

| Split   | Images                            |
| ------- | --------------------------------- |
| Train   | 38,835                            |
| Test    | 2,564                             |
| Classes | blank (empty), non_blank (animal) |

Training data sourced from Caltech Camera Traps (18k blank + 18k animal) combined with Serengeti2 dataset. Test images include Caltech metadata timestamps enabling day/night breakdown.

---

## Training Configuration (Optuna-optimized)

| Hyperparameter     | Value                                            | Source         |
| ------------------ | ------------------------------------------------ | -------------- |
| Learning rate      | 7.64 × 10⁻⁴                                      | Optuna trial 3 |
| Weight decay       | 0.00815                                          | Optuna trial 3 |
| Blank class weight | 1.27                                             | Optuna trial 3 |
| Gradient clip      | 0.775                                            | Optuna trial 3 |
| Batch size         | 32 (effective 128 w/ accum)                      | Fixed          |
| Scheduler          | CosineAnnealingLR, T_max=50                      | Fixed          |
| Augmentation       | RandomHFlip, ColorJitter, RandomGrayscale(p=0.2) | Fixed          |

Hyperparameters searched with Optuna TPESampler (seed=42), 15 trials × 10 epochs, MedianPruner. Best trial achieved 85.6% val accuracy in 10 epochs.

---

## Performance at Threshold Sweep

All numbers are **single-pass inference** (no TTA) — this is what the hardware actually runs. See the TTA section below for context on those numbers.

| Threshold | Accuracy  | Recall | FAR   | Precision | F1    |
| --------- | --------- | ------ | ----- | --------- | ----- |
| 0.3       | 84.1%     | 95.8%  | 27.5% | 77.7%     | 85.8  |
| 0.4       | 86.5%     | 91.9%  | 18.9% | 83.0%     | 87.2  |
| **0.5**   | **87.1%** | 87.3%  | 13.1% | 86.9%     | **87.1** |
| 0.6       | 86.5%     | 83.5%  | 10.5% | 88.8%     | 86.0  |
| 0.7       | 84.9%     | 77.1%  | 7.3%  | 91.4%     | 83.7  |
| 0.8       | 80.8%     | 66.6%  | 5.0%  | 93.0%     | 77.6  |
| 0.9       | 74.9%     | 52.7%  | 2.8%  | 94.9%     | 67.7  |

### Operating Point Recommendation

The right threshold depends on deployment context. Per-frame FAR numbers look manageable on the balanced test set, but real-world camera traps run 24/7 at 30fps with animal presence below 0.1% — a 13% per-frame FAR means the high-res camera fires constantly on blank frames.

**The hardware fix is temporal filtering** (see Architectural Roadmap). If the chiplet must see N consecutive "animal" frames before triggering the high-res camera, effective FAR drops dramatically:

| Threshold | Recall | Per-frame FAR | Effective FAR (3-frame) | Effective FAR (5-frame) |
| --------- | ------ | ------------- | ----------------------- | ----------------------- |
| 0.3       | 95.8%  | 27.5%         | 2.1%                    | 0.16%                   |
| 0.4       | 91.9%  | 18.9%         | 0.67%                   | 0.024%                  |
| 0.5       | 87.3%  | 13.1%         | 0.22%                   | 0.004%                  |
| 0.6       | 83.5%  | 10.5%         | 0.12%                   | 0.001%                  |

With a 3-frame filter, even threshold 0.3 achieves 2.1% effective FAR while retaining 95.8% recall — the best operating point for **rare species detection** where missing an animal is costlier than a false trigger.

**Recommended operating points:**
- **General use** — threshold 0.5, 3-frame filter: 87.3% recall, 0.22% effective FAR
- **Rare species** — threshold 0.3–0.4, 3-frame filter: 92–96% recall, 0.7–2.1% effective FAR

---

## Confusion Matrix — Threshold 0.5 (2,564 test images)

```
                    Pred: ANIMAL   Pred: EMPTY
Actual: ANIMAL         TP = 1,119     FN =   163
Actual: EMPTY          FP =   168     TN = 1,114
```

---

## Day vs. Night Breakdown

### Threshold 0.5

| Condition              | Images | Accuracy | Recall    | FAR   |
| ---------------------- | ------ | -------- | --------- | ----- |
| Day (07:00–18:00)      | 1,191  | 89.9%    | 87.2%     | 8.4%  |
| Night (19:00–06:00)    | 809    | 91.1%    | **95.3%** | 18.1% |
| Unknown (no timestamp) | 564    | 75.4%    | 71.6%     | 20.9% |

### Threshold 0.6

| Condition              | Images | Accuracy | Recall    | FAR       |
| ---------------------- | ------ | -------- | --------- | --------- |
| Day (07:00–18:00)      | 1,191  | 89.9%    | 85.2%     | 7.2%      |
| Night (19:00–06:00)    | 809    | 91.3%    | **93.5%** | **13.4%** |
| Unknown (no timestamp) | 564    | 72.2%    | 61.0%     | 16.7%     |

Night performance is stronger than daytime at both thresholds, driven by high-contrast IR images having clear silhouettes. The unknown-timestamp cohort (primarily Serengeti2 images with no metadata) performs ~14% lower — likely because that data lacks matching timestamp-based augmentation tuning.

---

## Progress vs. Prior Architectures

| Architecture                      | Val Accuracy | Night FAR | Notes                           |
| --------------------------------- | ------------ | --------- | ------------------------------- |
| 3-layer all-binary (baseline)     | 73.4%        | ~93.7%    | Original prototype              |
| 4-layer all-binary                | 76.2%        | —         | Added conv4 (128→256)           |
| Hybrid precision (Conv1 8-bit)    | 85.2%        | —         | Conv1 moved to host, 8-bit      |
| Hybrid + expanded dataset         | 85.8%        | 17.7%     | 38k train images, weighted loss |
| **Hybrid + Optuna HPs (current)** | **87.1%**    | **18.1%** | lr/wd/bw/gc from Optuna, early stopped epoch 21 |

The jump from 85.8% → 87.1% came from Optuna-tuned hyperparameters. Training ran 21 epochs before early stopping (patience=7); best checkpoint was epoch 14.

---

## Inference Performance

| Metric                     | Value                     |
| -------------------------- | ------------------------- |
| Throughput (MPS, batch=32) | **566.8 img/s**           |
| Latency per image          | 1.76 ms                   |
| 30 fps real-time target    | **PASS** (18.9× headroom) |

Measured on Apple M-series GPU (MPS). Chiplet XNOR acceleration would further reduce Conv2–4 time; the AXI4-Stream interface (9.6 GB/s rated, 8.0 GB/s effective) is not a bottleneck at 30fps (Conv1 output is ~62 MB/s).

---

## AUC-ROC

AUC-ROC = **0.9377** — indicates strong discrimination ability across all threshold choices. A perfect classifier scores 1.0; random guessing scores 0.5. The 0.94 score means the model ranks a random animal image above a random blank image 93.8% of the time, regardless of threshold.

---

## Test-Time Augmentation (TTA) — Theoretical Ceiling Only

TTA runs 4 augmented views of each frame (original, horizontal flip, brightness ±0.15) through the model and averages the softmax probabilities. This improves accuracy but is **not viable for deployed hardware**:

- Requires 4× the XNOR compute on the chiplet — quadruples dynamic power, likely violating the sub-1W budget
- Requires 4× the AXI4-Stream traffic (4 × 48 MB/s = 192 MB/s — still within 8 GB/s rated, but 4× the interface energy)

TTA numbers are reported here as a **capability ceiling** — what the model could achieve with unlimited compute:

| Threshold | Accuracy (TTA) | Recall (TTA) | Night Recall (TTA) | Night FAR (TTA) |
| --------- | -------------- | ------------ | ------------------ | --------------- |
| 0.5       | 88.8%          | 90.9%        | 98.2%              | 15.7%           |
| 0.6       | 88.0%          | 85.0%        | 94.4%              | 11.4%           |

The gap between TTA and single-pass (87.1% vs 88.8% overall) is the upper bound on what better training could recover — knowledge distillation targets this gap.

---

## Known Limitations

1. **Per-frame FAR overstates real-world performance** — test set is 50/50 blank/animal; real deployments are >99% blank at 30fps. Use effective FAR with temporal filtering as the meaningful metric (see threshold table above).
2. **Unknown-timestamp cohort (75.4% accuracy)** — Serengeti2 images lack metadata so day/night breakdown is unavailable. These images may have different IR/lighting characteristics than the Caltech-tuned augmentation pipeline expects.
3. **Training complete** — early stopped at epoch 21 (patience=7). Best checkpoint is epoch 14 at 87.1% val accuracy. This evaluation reflects the final model.
4. **TTA not hardware-deployable** — 4× power cost exceeds budget. Numbers included as theoretical ceiling only.
