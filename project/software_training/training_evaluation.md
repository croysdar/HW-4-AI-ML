# BNN Accelerator Chiplet — Model Evaluation Report

ECE 510 Spring 2026  
Dataset: Caltech Camera Traps + Serengeti2 (combined)  
Evaluation date: 2026-04-24

---

## Model Summary

| Property                           | Value                                    |
| ---------------------------------- | ---------------------------------------- |
| Architecture                       | Hybrid-precision BNNClassifier (4-layer) |
| Parameters                         | 389,410                                  |
| Checkpoint size                    | 4.70 MB                                  |
| Best val accuracy — baseline       | 87.1% (epoch 14 of 21, early stopped)   |
| Best val accuracy — distilled      | 87.6% (epoch 23 of 30, early stopped)   |
| Best eval accuracy — ensemble      | **88.6%** (baseline + distilled averaged) |
| AUC-ROC (baseline)                 | **0.9377**                               |

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

Conv1 runs on the host ARM CPU and outputs 8-bit feature maps (32×224×224) over AXI4-Stream. Conv2–Conv4 execute on the chiplet as 1-bit XNOR+Popcount operations. The distilled checkpoint uses the identical architecture — only the training procedure differs.

---

## Dataset

| Split   | Images                            |
| ------- | --------------------------------- |
| Train   | 38,835                            |
| Test    | 2,564                             |
| Classes | blank (empty), non_blank (animal) |

Training data sourced from Caltech Camera Traps (18k blank + 18k animal) combined with Serengeti2 dataset, all at 224×224. Test images include Caltech metadata timestamps enabling day/night breakdown.

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

All numbers are **single-pass inference** (no TTA) on the distilled model checkpoint. Baseline (non-distilled) numbers shown where distilled was not evaluated.

### Distilled Model (bnn_distilled.pth)

| Threshold | Accuracy  | Recall    | FAR   | Precision |
| --------- | --------- | --------- | ----- | --------- |
| **0.5**   | **87.6%** | **90.2%** | 15.1% | 85.7%     |
| 0.6       | 87.3%     | 87.1%     | 12.6% | 87.4%     |
| 0.7       | 86.2%     | 83.3%     | 10.9% | 88.4%     |

### Baseline Model (bnn_serengeti2.pth) — for reference

| Threshold | Accuracy  | Recall | FAR   | Precision | F1   |
| --------- | --------- | ------ | ----- | --------- | ---- |
| 0.3       | 84.1%     | 95.8%  | 27.5% | 77.7%     | 85.8 |
| 0.4       | 86.5%     | 91.9%  | 18.9% | 83.0%     | 87.2 |
| **0.5**   | **87.1%** | 87.3%  | 13.1% | 86.9%     | 87.1 |
| 0.6       | 86.5%     | 83.5%  | 10.5% | 88.8%     | 86.0 |
| 0.7       | 84.9%     | 77.1%  | 7.3%  | 91.4%     | 83.7 |
| 0.8       | 80.8%     | 66.6%  | 5.0%  | 93.0%     | 77.6 |
| 0.9       | 74.9%     | 52.7%  | 2.8%  | 94.9%     | 67.7 |

### Operating Point Recommendation

The right threshold depends on deployment context. Per-frame FAR numbers look manageable on the balanced test set, but real-world camera traps run 24/7 at 30fps with animal presence below 0.1% — a 15% per-frame FAR means the high-res camera fires constantly on blank frames.

**The hardware fix is temporal filtering** (see Architectural Roadmap). If the chiplet must see N consecutive "animal" frames before triggering the high-res camera, effective FAR drops dramatically:

#### Distilled model with temporal filtering

| Threshold | Recall | Per-frame FAR | Effective FAR (3-frame) | Effective FAR (5-frame) |
| --------- | ------ | ------------- | ----------------------- | ----------------------- |
| 0.5       | 90.2%  | 15.1%         | 0.34%                   | 0.008%                  |
| 0.6       | 87.1%  | 12.6%         | 0.20%                   | 0.003%                  |
| 0.7       | 83.3%  | 10.9%         | 0.13%                   | 0.002%                  |

**Recommended operating points (distilled model):**
- **General use** — threshold 0.5, 3-frame filter: 90.2% recall, 0.34% effective FAR
- **Rare species** — threshold 0.5, 5-frame filter: 90.2% recall, 0.008% effective FAR

---

## Confusion Matrix — Threshold 0.5 (ensemble, 2,564 test images)

```
                    Pred: ANIMAL   Pred: EMPTY
Actual: ANIMAL         TP = 1,155     FN =   127
Actual: EMPTY          FP =   166     TN = 1,116
```

---

## Day vs. Night Breakdown (Ensemble — greyscale IR labels)

Day/night classification uses the greyscale IR detector (`label_ir_images.py`) as the primary signal — the physical ground truth for whether the camera was in IR mode. Timestamps are used as fallback only. Cross-validation against Caltech timestamps found 10.7% disagreement (1,915 images the clock called "day" were IR; 449 called "night" were colour), confirming timestamps alone are unreliable.

### Threshold 0.5

| Condition             | Images | Accuracy | Recall    | FAR   |
| --------------------- | ------ | -------- | --------- | ----- |
| Day (colour, IR off)  | 1,509  | 85.2%    | 79.5%     | 11.5% |
| Night (IR active)     | 1,055  | 93.4%    | **98.0%** | 17.2% |

### Threshold 0.6

| Condition             | Images | Accuracy | Recall    | FAR   |
| --------------------- | ------ | -------- | --------- | ----- |
| Day (colour, IR off)  | 1,509  | 85.5%    | 72.9%     | 7.4%  |
| Night (IR active)     | 1,055  | 92.5%    | **96.1%** | 15.7% |

### Threshold 0.7

| Condition             | Images | Accuracy | Recall    | FAR   |
| --------------------- | ------ | -------- | --------- | ----- |
| Day (colour, IR off)  | 1,509  | 83.8%    | 64.1%     | 5.1%  |
| Night (IR active)     | 1,055  | 90.7%    | **92.9%** | 14.4% |

Night (IR) performance is consistently stronger: 98% recall at threshold 0.5 vs 79.5% daytime. High-contrast IR silhouettes are inherently easier to classify than colour images with complex backgrounds. The previously reported "day=92.2%" figure (timestamp-based) was inflated by excluding 564 hard Serengeti images; 85.2% is the corrected number. No "unknown" cohort remains — all 2,564 test images are classified by the greyscale detector.

---

## Progress vs. Prior Architectures

| Architecture                          | Val Accuracy | Night FAR | Notes                                        |
| ------------------------------------- | ------------ | --------- | -------------------------------------------- |
| 3-layer all-binary (baseline)         | 73.4%        | ~93.7%    | Original prototype                           |
| 4-layer all-binary                    | 76.2%        | —         | Added conv4 (128→256)                        |
| Hybrid precision (Conv1 8-bit)        | 85.2%        | —         | Conv1 moved to host, 8-bit                   |
| Hybrid + expanded dataset             | 85.8%        | 17.7%     | 38k train images, weighted loss              |
| Hybrid + Optuna HPs (baseline)        | 87.1%        | 18.1%     | lr/wd/bw/gc from Optuna, early stopped ep 21 |
| Distilled (ResNet-50 → BNN)           | 87.6%        | 17.2%     | KD T=2, α=0.3, ResNet-50 teacher at 94.6%   |
| **Ensemble (baseline + distilled)**   | **88.6%**    | **17.2%** | Runtime average of both checkpoints          |

Night FAR column now uses greyscale IR labels (corrected from timestamp-based). The ensemble achieves the best overall accuracy; night recall is 98.0% at threshold 0.5 across both distilled and ensemble.

---

## Knowledge Distillation

Two-phase distillation: ResNet-50 teacher fine-tuned on the same dataset, then BNNClassifier student trained with combined KL-divergence + cross-entropy loss.

| Setting          | Value                                              |
| ---------------- | -------------------------------------------------- |
| Teacher          | ResNet-50 (ImageNet pretrained, fine-tuned 3 epochs) |
| Teacher val acc  | 94.6%                                              |
| Temperature      | 2.0                                                |
| Alpha (KL weight)| 0.3 (hard CE weight = 0.7)                         |
| Student epochs   | 30 (early stopped)                                 |

**Attempts and findings:**
- ResNet-18 teacher (89.4%) with α=0.7, T=4: student peaked at **83.6%** — worse than baseline. Gap too small, KL dominated.
- ResNet-50 teacher (94.6%) with α=0.7, T=4: student peaked at **84.9%** — still worse. Same issue.
- ResNet-50 teacher (94.6%) with α=0.3, T=2: student peaked at **87.6%** — beats baseline by 0.5%.

Letting hard CE dominate (α=0.3) was the key fix. With a large architectural gap between float32 ResNet-50 and 1-bit BNN, soft labels alone don't transfer well — the student benefits more from ground-truth labels with a small regularizing nudge from the teacher's distributions.

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

AUC-ROC = **0.9377** (baseline model). The distilled model is expected to be marginally higher given the improved recall; full AUC sweep not yet run on distilled checkpoint.

---

## Test-Time Augmentation (TTA) — Theoretical Ceiling Only

TTA runs 4 augmented views of each frame through the model and averages softmax probabilities. This is **not viable for deployed hardware**:

- Requires 4× the XNOR compute on the chiplet — quadruples dynamic power, likely violating the sub-1W budget
- Requires 4× the AXI4-Stream traffic

TTA numbers are reported here as a **capability ceiling** — what the model could achieve with unlimited compute:

| Threshold | Accuracy (TTA) | Recall (TTA) | Night Recall (TTA) | Night FAR (TTA) |
| --------- | -------------- | ------------ | ------------------ | --------------- |
| 0.5       | 88.8%          | 90.9%        | 98.2%              | 15.7%           |
| 0.6       | 88.0%          | 85.0%        | 94.4%              | 11.4%           |

---

## Temporal Filter Evaluation on Real Sequences

Downloaded 86 complete 5-frame sequences (43 blank, 43 animal) from Caltech Camera Traps using `download_sequences.py`, ensuring zero overlap with the training set (images excluded by seed-matched reconstruction of the download script's sampling). Evaluated with `evaluate_sequences.py` — ensemble model, threshold 0.5, 3-frame filter.

### Results

| Class  | Sequences | Per-frame FAR/Recall | Sequence FAR/Recall |
| ------ | --------- | -------------------- | ------------------- |
| Blank  | 43        | **96.7% FAR**        | **100% FAR**        |
| Animal | 43        | ~88% recall          | ~91% recall         |

The animal recall matched expectations. The blank FAR is anomalously high — 96.7% per-frame vs the 11–15% seen on the random test set.

### Root Cause: PIR Selection Bias in Multi-Frame Blanks

A sanity check confirmed inference code is correct:

| Pool            | n   | Mean p(animal) | FAR @ 0.5 |
| --------------- | --- | -------------- | --------- |
| Random test blanks | 20 | 0.238 | ~20% |
| 5-frame blank sequences | 215 | **0.808** | **96.7%** |

The 5-frame blank sequences are **not a random sample of blank scenes** — they are scenes where the PIR motion sensor fired on 5 consecutive frames. A PIR sensor fires on persistent movement: wind-blown vegetation, grass, water ripples, shadows. These are exactly the scenes most likely to fool a visual classifier. Multi-frame blank sequences are selection-biased toward the hardest possible blank examples, while random test-set blanks are an unbiased sample from the full distribution.

This also breaks the temporal filter's independence assumption. The effective FAR calculation (FAR^N assumes i.i.d. frames) does not hold for persistent false-alarm scenes: if frame 1 fools the model, frames 2–5 in the same scene will also fool it, so the filter provides little protection.

### Implications and Options

The theoretical FAR^3 = 0.34% (threshold 0.5) is optimistic — real-world scenes with persistent motion will have near-100% per-frame FAR on all frames in the window, making the temporal filter ineffective for that class of scene.

Options to address this:

| Option | Description | Tradeoff |
| ------ | ----------- | -------- |
| **Hard-blank retraining** | Download more multi-frame blank sequences; use them as training negatives. Forces model to learn that persistent background motion is blank. | Requires multi-frame training pipeline; unclear how many are needed. |
| **Scene-level augmentation** | Apply temporal crops/jitter during training to simulate persistent scenes | Simpler; may not capture real PIR-bias distribution. |
| **Adaptive threshold** | Raise threshold to 0.7–0.8 for high-motion scenes (detected by optical flow or motion magnitude) | Adds complexity; may miss low-contrast animals in high-motion scenes. |
| **Accept the limitation** | Document that temporal filter is not the right tool for persistent-motion backgrounds; hardware should also gate on motion delta (frame difference) | Simplest; honest about model scope. |

The recommended near-term fix is to retrain with hard-blank sequences added to the training negatives. The download infrastructure (`download_sequences.py`) already exists — retrain with `--n 200` blank sequences interleaved with the existing training set.

---

## Known Limitations

1. **Per-frame FAR overstates real-world performance** — test set is 50/50 blank/animal; real deployments are >99% blank at 30fps. Use effective FAR with temporal filtering as the meaningful metric.
2. **Temporal filter independence assumption breaks for persistent-motion blanks** — 5-frame blank sequences are PIR-selected toward hard scenes (wind, vegetation, shadows); model consistently assigns p(animal)≈0.8 on all 5 frames, so FAR^N is far too optimistic for this sub-class of blanks.
3. **Unknown-timestamp cohort** — Serengeti2 images lack metadata; day/night figures from the greyscale IR classifier are the reliable signal.
4. **Distillation ceiling** — the +0.5% gain from distillation reflects the architectural capacity gap between float32 ResNet-50 and 1-bit BNN. Further gains would require feature-level distillation (FitNets) rather than output-level soft labels.
5. **TTA not hardware-deployable** — 4× power cost exceeds budget. Numbers included as theoretical ceiling only.
