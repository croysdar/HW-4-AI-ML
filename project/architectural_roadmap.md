# Architectural Roadmap & Future Work

**Project:** Ultra-Low-Power BNN Wildlife Smart Filter  
**Course:** ECE 510 - Hardware for AI  
**Author:** Rebecca Gilbert-Croysdale

---

## 1. Current Status

The design uses a **hybrid-precision 4-layer BNN** trained on Caltech Camera Traps + Serengeti2 (38,835 train images).

| Metric | Value |
|---|---|
| Val accuracy | **87.1%** (best epoch 14/21, early stopped) |
| AUC-ROC | 0.9377 |
| Night recall | 95.3% @ threshold 0.5 |
| Night FAR | 13.4% @ threshold 0.6 (recommended) |
| Inference throughput | 566 img/s (1.76 ms/img on MPS) |

**Interface:** 256-bit AXI4-Stream @ 300 MHz (9.6 GB/s rated, 8.0 GB/s effective)  
**Compute:** 1-bit XNOR-Popcount engine (Conv2–4 on chiplet)  
**Partition:** Conv1 (8-bit) on host ARM CPU; Conv2–4 (1-bit) on chiplet XNOR engine

### Architecture progression

| Version | Val Accuracy | Change |
|---|---|---|
| 3-layer all-binary | 73.4% | Original prototype |
| 4-layer all-binary | 76.2% | Added Conv4 (128→256 channels) |
| Hybrid precision | 85.2% | Conv1 retained as 8-bit on host |
| + Expanded dataset + weighted loss | 85.8% | 38k images, blank weight=1.5 |
| **+ Optuna HPs (current)** | **87.1%** | lr/wd/blank_weight/grad_clip tuned |

---

## 2. Software Improvements

### 2.1 ~~Hybrid Precision Implementation~~ ✅ DONE

Completed. Conv1 retained at 8-bit (nn.Conv2d on host CPU); Conv2–4 are 1-bit BinarizeConv2d on chiplet. Conv4 widened to 256 channels. Accuracy improved from 73.4% → 87.1%.

The chiplet remains purely 1-bit — no mixed-precision MAC array needed on chip.

### 2.1.1 Architecture Decision Record: INT8 Layer Stays on Host CPU

**Decision (M2, 2026-05-02):** Conv1 will remain on the ARM Host CPU as an INT8 fixed-point layer. It will **not** be moved onto the custom BNN chiplet.

**Hardware rationale:**

1. **Silicon area explosion.** A full INT8 MAC unit (8-bit × 8-bit → 16-bit accumulator) requires roughly 4–8× the transistor area of a 1-bit XNOR gate. Placing even one INT8 MAC array on the chiplet would require a distinct datapath alongside the existing XNOR+Popcount engine. For a 256-wide vector unit operating at 300 MHz, an INT8 lane array would consume on the order of 50–100 kGE (gate equivalents) — a significant fraction of the total chiplet area budget for a die designed around ultra-minimal 1-bit logic.

2. **Dark silicon / underutilization.** The chiplet executes Conv1, Conv2, Conv3, Conv4 sequentially. If an INT8 MAC array were included on chip, it would be active only during the Conv1 phase (~17% of total per-layer compute time, since Conv1 is 224×224 vs. Conv2–4 at 112×112, 56×56, 28×28 respectively). The INT8 array would sit dark and leaking power during the three 1-bit layers that dominate inference time. Dark silicon at constrained power budgets is an unacceptable waste.

3. **SystemVerilog control simplicity.** The existing `bnn_conv_core.sv` implements a single, uniform 256-bit XNOR+Popcount datapath. Adding INT8 support would require a second control FSM (or a mode-switched FSM), separate precision-aware address generation, and an INT8-to-1-bit format conversion at the Conv1/Conv2 boundary. Keeping INT8 on the host CPU means the chiplet control logic remains a single-mode streaming engine — cleaner RTL, faster timing closure, fewer verification corner cases.

**Validation (M2):** Fake-quantization experiments (`project/m2/validate_precision.py`) confirm an accuracy delta of −4.0% and logit MAE of 0.2453 for INT8 Conv1 vs. FP32 Conv1 on 100 test images. This cost is acceptable given the solar power constraint and high-recall mission objective (see `project/m2/precision.md`).

### 2.2 ~~Optuna Hyperparameter Search~~ ✅ DONE

Completed. 15 trials × 10 epochs. Best: lr=7.64e-4, weight_decay=0.00815, blank_weight=1.27, grad_clip=0.775. Gained ~1.3% accuracy over hand-tuned defaults.

### 2.3 Test-Time Augmentation (TTA) — quantify gain

TTA is implemented in `confidence_check()` (4 views: original, H-flip, ±brightness). Not yet run through `evaluate_bnn.py` to measure the actual accuracy delta.

**Effort:** Low — add `--tta` flag to evaluate_bnn.py, re-run evaluation.  
**Expected gain:** 0.5–1% accuracy, modest FAR reduction.  
**Worth doing:** Yes, especially if night FAR remains a concern.

### 2.4 Dataset Expansion — WCS Camera Traps

WCS Camera Traps has 1.37M images (591k blank, 786k animal) but **no timestamps**, so day/night filtering is not possible. Decided to defer until weighted loss + current data were fully evaluated.

**Current state:** 87.1% appears to have plateaued (early stopped). Additional data may help push past this ceiling.  
**Consideration:** WCS images skew toward daytime wildlife; adding them may improve overall recall but not specifically night FAR.  
**Recommendation:** Try after TTA is quantified — if accuracy is still stuck at ~87%, add WCS and retrain.

### 2.5 Knowledge Distillation (longer-term)

Train a larger FP32 "teacher" network to convergence, then distill soft targets into the BNN "student." BNNs typically gain 2–5% accuracy from distillation vs. training from scratch with hard labels.

**Effort:** Medium — requires implementing distillation loss (KL divergence on soft outputs).  
**Expected gain:** 2–4% on top of current 87.1%, potentially reaching ~90%+.  
**Worth doing:** Yes, if the project requires pushing accuracy higher and dataset expansion alone is insufficient.

---

## 3. Hardware Improvements

### 3.1 Temporal Filtering — Consecutive-Frame Trigger

The per-frame FAR from the BNN (~13% at threshold 0.5) is too high for 24/7 continuous operation at 30fps — real animal presence can be below 0.02%, so false alarms dominate triggers badly.

**Fix:** Add a small counter in SystemVerilog that only asserts the high-res camera trigger after N consecutive "animal" frames from the chiplet. A random noise false alarm must persist across N independent frames, which is exponentially unlikely.

**Math:**
- Per-frame FAR = 13.1% at threshold 0.5
- 3-frame effective FAR = 0.131³ = **0.22%**
- 5-frame effective FAR = 0.131⁵ = **0.004%**
- An actual animal at 30fps stays in frame for 60+ frames — easily clears any N threshold

**Implementation:** A 2-bit saturating counter in the chiplet control logic. Cost: ~10 LUTs. Power impact: negligible.

**Impact on operating point:** With a 3-frame filter, threshold can be lowered to 0.3–0.4 to maximize recall (91–96%) while keeping effective FAR below 2%. This is the right configuration for rare species detection.

**Status:** Architectural design — not yet implemented in RTL.

### 3.3 Weight-Stationary Dataflow

Load weights into local scratchpad SRAM on the chiplet once per inference; stream input feature maps from host over AXI. This eliminates repeated weight fetches from DRAM during convolution.

**Impact:** Reduces AXI bus traffic significantly; DRAM power drops for batch or burst workloads.  
**Status:** Architectural design only — not yet implemented in RTL.

### 3.4 SRAM-Based Computing-In-Memory (CIM)

Modify SRAM bitcell sense-amplifiers to perform XNOR logic internally. Weights never leave the memory array.

**Impact:** Eliminates ~90% of internal data movement power.  
**Tools:** Cadence Virtuoso, Synopsys (full-custom flow, outside OpenLane 2 digital flow).  
**Status:** Future research direction.

---

## 4. Long-Term Research: Non-Volatile Logic

The ultimate evolution involves **Memristor Crossbar Arrays** for in-memory analog BNN inference.

- Weights stored non-volatilely with zero leakage power
- Multiplication via Ohm's Law, addition via Kirchhoff's Current Law
- Outside current OpenLane 2 digital flow; represents the 100× efficiency leap needed for multi-year solar-harvesting edge deployment

---

## 5. Recommended Next Steps (priority order)

1. **RTL temporal filtering** — ~10 LUTs, negligible power, crushes real-world FAR. Most impactful hardware addition before the defense.
2. **Knowledge distillation** — teacher (ResNet-18) training in progress; student distillation follows. Targets ~90%+ accuracy with identical hardware.
3. **Add WCS data and retrain** — if accuracy ceiling persists after distillation.
4. **RTL weight-stationary dataflow** — reduces AXI traffic and DRAM power for burst workloads.

### Note on TTA

TTA is **not recommended for hardware deployment** — 4× XNOR compute quadruples chiplet dynamic power, likely exceeding the sub-1W budget. TTA numbers (88.8% overall, 98.2% night recall) are reported in the evaluation report as a theoretical ceiling only. The deployed system uses single-pass inference with temporal filtering to manage FAR instead.
