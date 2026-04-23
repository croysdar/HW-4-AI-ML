# Documentation Updates Required

ECE 510 Spring 2026 — Tracking changes needed after hybrid precision architecture update.

---

## High Priority — Diagrams (require redraw)

### algorithm_diagram.png
**What's wrong:**
- Shows "BinarizeConv2d Engine (3× Conv Layers)" — now 4 conv layers total
- Shows "1-bit Input Tensors" going into chiplet — now 8-bit feature maps from Conv1 (host) feed into chiplet
- "On-Chip SRAM (1-bit weights & activations)" — Conv1 outputs are 8-bit, not 1-bit

**What it should show:**
- Host CPU runs Conv1 (8-bit) and sends 8-bit feature maps over AXI
- Chiplet XNOR engine runs Conv2, Conv3, Conv4 (all 1-bit)
- AXI label: "8-bit Feature Maps (Conv1 output)"
- Engine label: "BinarizeConv2d Engine (3× Conv Layers, 1-bit)"

---

### system_diagram.png
**What's wrong:**
- Host CPU box lists "Binarization" as a host task — Conv1 is now 8-bit conv on host, not binarization
- Arrow label "1-bit Input Tensors" is incorrect — chiplet receives 8-bit activations from Conv1

**What it should show:**
- Host CPU box: replace "Binarization" with "Conv1 (8-bit), BatchNorm"
- AXI arrow label: "8-bit Activations (32×224×224)"
- Chiplet SRAM: "1-bit weights & activations (Conv2–4)"

---

## Medium Priority — Text Documents

### partition_rationale.md
**What's wrong:**
- "All three BinarizeConv2d layers moved to hardware" — now 3 BNN layers on chiplet + Conv1 (8-bit) on host
- Runtime bottleneck analysis references 3-layer all-binary model

**What to update:**
- Section 2: "Conv1 (8-bit std Conv2d) retained on host CPU — output feeds chiplet as 8-bit feature map"
- Section 2: "Conv2, Conv3, Conv4 (1-bit BinarizeConv2d) moved to chiplet XNOR engine"
- Note that the hybrid partition keeps the chiplet purely binary (no 8-bit MAC array needed on chip)

---

### heilmeier.md
**What's wrong:**
- "Reducing memory traffic by a factor of 32" — Conv1 is 8-bit (4x reduction vs FP32), not 32x. Conv2–4 are still 32x.
- "Arithmetic intensity from 12.34 FLOP/byte to nearly 395 FLOP/byte" — hybrid AI is ~572 FLOP/byte overall

**What to update:**
- Question 3: Soften "factor of 32" to "up to 32x for the binary layers, with the first layer retained at 8-bit to preserve spatial fidelity"
- AI figure: update to reflect hybrid operating point

---

## Low Priority — Optional / Presentational

### roofline_project.png
**What's wrong:**
- Target HW point (150 FLOP/byte) was for pure BNN — hybrid has higher AI (~572 FLOP/byte)
- Still tells the correct story (memory-bound → compute-bound) but target point is conservative

**What to update (if regenerating):**
- Add a second target point for the hybrid architecture at ~572 FLOP/byte
- Or update annotation on existing point to note it is a conservative lower bound

---

## No Changes Needed

| Document | Reason |
|---|---|
| sw_baseline.md | Intentional snapshot of original 3-layer CPU baseline — correct as-is |
| ai_calculation.md | CPU SW baseline calculation — still valid as problem motivation |
| interface_selection.md | 9.6 GB/s interface still sufficient; hybrid reduces required BW further |
| power_estimate.md | Updated April 2026 to reflect hybrid partition |

---

## Planned Algorithm Improvements (not yet implemented)

### Test-Time Augmentation (TTA)
At inference, run each image through the model multiple times with small augmentations (horizontal flip, slight brightness shift) and average the softmax probabilities before thresholding. No retraining needed. Implement in `confidence_check()` in `bnn_serengeti2.py`. Expected gain: 0.5–1% accuracy.

### Optuna Hyperparameter Search
After weighted loss and TTA are locked in, run an Optuna study to tune: LR, weight decay, ACCUM_STEPS, cosine eta_min, and class weight ratio. Each trial ~8 min on current dataset. Run 20–30 trials overnight.

---

## Software / Code

| File | Status |
|---|---|
| bnn_serengeti2.py | Updated — hybrid architecture, 4 layers, tqdm, per-epoch Recall/FAR |
| evaluate_bnn.py | No changes needed |
| combine_datasets.py | No changes needed |
| download_lila_dataset.py | No changes needed |
