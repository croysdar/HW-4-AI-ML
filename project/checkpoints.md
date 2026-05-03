# Model Checkpoints

| File | Date | Val Acc | Notes |
|------|------|---------|-------|
| `bnn_4layer_76pct.pth` | Apr 22 | ~76% | Early 4-layer BNN prototype |
| `bnn_hybrid_85pct_V1.pth` | Apr 22 | ~85% | Hybrid architecture V1 |
| `bnn_teacher.pth` | Apr 23 | — | Full-precision ResNet teacher model for distillation |
| `bnn_distilled.pth` | Apr 24 | — | BNN trained via knowledge distillation (intermediate) |
| `bnn_distilled_876pct.pth` | Apr 24 | 87.6% | Best distilled BNN |
| `bnn_baseline_871pct.pth` | Apr 24 | 87.1% | Baseline BNN (bnn_serengeti2 architecture), no RRR |
| `bnn_framediff.pth` | Apr 24 | 68.8% | Frame-differencing temporal model, best was 68.8% seq acc |
| `bnn_serengeti2_coldstart_84p4.pth` | Apr 25 | 84.4% | Cold-start, top-7px + bottom-right corner masked, no RRR |
| `bnn_serengeti2.pth` | Apr 25 | in progress | Cold-start, uniform 5px border masked all edges, no RRR |
| `bnn_dualModel_1_day.pth` | May 2 | 84.7% | Day-only model; 21,076 train / 1,870 test (balanced); best epoch 34; recall 83.5% FAR 14.1% |
| `bnn_dualModel_1_night.pth` | May 2 | 83.2% | Night-only model; 16,515 train / 1,450 test (balanced); best epoch 35; recall 81.4% FAR 15.0%; class weights not adjusted for night imbalance — needs Optuna |

## Current best for deployment
`bnn_distilled_876pct.pth` — 87.6% val acc, full-precision teacher distilled into BNN.

## Notes
- All `bnn_serengeti2*` checkpoints use the 5-layer BNN architecture (Conv1 8-bit + 3× BinarizeConv2d 1-bit).
- Car images (410 stems) are blacklisted from training in all runs after Apr 24.
- Banner masking was added progressively: top-7px only → top-7px + bottom-right corner → uniform 5px all edges (current).

---

## Attempt: Environment-Triggered Weight Context Switching

**Date:** May 2, 2026
**Checkpoints:** `bnn_dualModel_1_day.pth`, `bnn_dualModel_1_night.pth`
**Hypothesis:** Training separate day and night models on their respective image distributions (color daytime vs. IR nighttime) should yield higher per-environment accuracy than a single combined model, because the feature distributions are fundamentally different.

**Day/Night Split:** Based on `ir_tod` labels in `ser_tod_labels.csv` (colourfulness score threshold ≥ 10.0 = day). Symlink datasets: `data_20k_day/` and `data_20k_night/`.

### Deployment Architecture: Weight-Stationary Dataflow

The camera has two operating modes — IR (night) and full-color (day) — and the hardware already signals this with a mode flip. The proposed deployment exploits that signal to swap weight context:

**Storage hierarchy:**

| Layer | Location | Contents |
|-------|----------|----------|
| Long-term | SD card / Flash | `day_weights.bin`, `night_weights.bin` (~50 KB each — negligible on any SD card) |
| Staging | System RAM (DDR) | Active weight file, loaded at boot or on mode change |
| Execution | On-chip SRAM (chiplet scratchpad) | Weights held stationary while image frames stream past |

**Context switch sequence (e.g., camera switches to night IR):**
1. Host ARM CPU detects IR mode active
2. CPU fetches `night_weights.bin` from SD card into DDR
3. CPU pushes weights to chiplet over AXI-Stream interface
4. Chiplet SRAM overwrites with new weights
5. 30 FPS inference resumes with night model — no further weight fetches until next mode change

**Weight-Stationary Dataflow:** Once weights are resident in chiplet SRAM, they stay fixed while activation data (image frames) streams through. This minimizes memory-wall pressure: weights are fetched from the "library" once per environment change (e.g., at sunset), not once per frame.

This satisfies M2 requirements: AXI-Stream as the interface protocol, 1-bit weight format keeping byte traffic minimal, and a roofline-justified memory hierarchy.

---

## Attempt: RRR (Right for the Right Reasons) — Post-Mortem

**Date:** May 2, 2026
**Status:** Abandoned. Officially moving forward with Heterogeneous INT8/1-bit design without spatial regularization.

### The Goal

The day model's occlusion maps showed the model was keying off background texture (left border, ground patches) rather than the animal. Quantified with a bbox alignment metric: only **6.7% of positive heatmap energy** fell inside the ground-truth bounding boxes (1/20 images ≥ 50% aligned). The hypothesis was that RRR — penalizing bn3 feature map activations outside the annotated bbox — would force the model to concentrate spatial attention on the animal and reduce the False Alarm Rate.

### The Failure

Three training strategies were attempted, all producing the same result:

| Strategy | RRR λ | Epochs | Alignment | Outcome |
|----------|--------|--------|-----------|---------|
| Resume from epoch 12 | 0.1 | 6 | ~6.7% (flat) | RRR loss 0.83 → 0.83, no movement |
| Warm-start from best ckpt | 0.3 | 3 | 11.5% | Confidence collapsed; FAR spiked to 14% |
| Fresh restart | 0.3 | 8 | ~11% | RRR loss flat, ~5 epochs slower to converge than baseline |

Alignment moved from **6.7% → 11.5%** across all strategies — marginal, and accompanied by significant confidence collapse. Images previously detected at p=0.74–0.98 fell to p=0.21–0.44 ("attention spreading"). The model's per-epoch RRR loss column held at ~0.83 across all runs regardless of λ or starting point, indicating the penalty was not driving spatial restructuring.

### Root Cause: 1-bit Weights Are Too Coarse for Spatial Penalties

The RRR penalty operates on ReLU-activated bn3 feature maps (56×56). In a standard floating-point network, a spatial gradient penalty can selectively suppress out-of-bbox channels while strengthening in-bbox ones. In a BNN, weights are constrained to {+1, −1} — there is no intermediate representation. The path of least resistance for minimizing `outside_attention / total_attention` is to globally suppress the entire channel's activation magnitude rather than redirect it spatially. This "blinds" the channel uniformly, which explains the confidence collapse without alignment improvement.

### The Evidence

Occlusion maps (patch=32, stride=16) run before and after RRR showed max_drop values increasing (0.003–0.006 → 0.010–0.032), meaning the model became *more* reactive to patch occlusion overall, but the hotspots remained off the animal — consistent with distributed suppression rather than spatial concentration. Grad-CAM attention maps confirmed hyper-sensitivity to random background patches post-RRR. The bbox alignment metric (fraction of positive heatmap energy inside ground-truth bbox) was the quantitative measure used throughout.

### Decision

RRR is not a suitable regularization tool for binary neural networks. The 1-bit weight constraint prevents the fine-grained spatial gradient updates that RRR requires. Going forward: train without spatial regularization, accept that BNN attention is diffuse, and rely on classification accuracy, recall, and FAR as the primary metrics.
