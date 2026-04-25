# Arithmetic Intensity Calculation — Updated (Hybrid Architecture)

ECE 510 Spring 2026 — Hybrid Precision BNNClassifier (4-layer)

> **Version note:** This updates `ai_calculation.md` for the hybrid-precision architecture.
> The original (3-layer all-binary, M1) is preserved in `ai_calculation.md`.

---

## 1. Architecture Change Summary

The original 3-layer model used `BinarizeConv2d` for all three conv layers including conv1.
The updated model introduces a **hybrid partition**:

| Layer | Type | Precision | Partition |
|---|---|---|---|
| conv1 (3→32, 224×224, s=1) | nn.Conv2d | 8-bit | Host ARM CPU |
| conv2 (32→64, 224→112, s=2) | BinarizeConv2d | 1-bit | Chiplet XNOR |
| conv3 (64→128, 112→56, s=2) | BinarizeConv2d | 1-bit | Chiplet XNOR |
| conv4 (128→256, 56→28, s=2) | BinarizeConv2d | 1-bit | Chiplet XNOR |

The first layer is retained at 8-bit to preserve spatial fidelity at 224×224 — binarizing the raw 3-channel RGB input directly causes significant information loss. Conv1 output (32×224×224) is transferred to the chiplet over AXI4-Stream as 8-bit activations.

---

## 2. Per-Layer FLOPs and Arithmetic Intensity (SW float32 baseline)

Using the standard no-reuse assumption (all weights, inputs, outputs touch DRAM once) with FP32 (4 bytes/element).

```
MACs  = C_out × H_out × W_out × C_in × K²
FLOPs = 2 × MACs
```

### conv1 — Host CPU (8-bit, but SW baseline runs FP32)

| Symbol | Value |
|---|---|
| C_in / C_out | 3 / 32 |
| H_out / W_out | 224 / 224 (stride=1, pad=1) |
| K | 3 |

```
MACs  = 32 × 224 × 224 × 3 × 9 = 43,352,064
FLOPs = 86,704,128

Bytes = (32×3×9 + 3×224×224 + 32×224×224) × 4
      = (864 + 150,528 + 1,605,632) × 4
      = 7,028,096 B  (6.7 MB)

AI = 86,704,128 / 7,028,096 = 12.3 FLOP/byte   ← same as original
```

### conv2 — Chiplet (SW: FP32, HW: 1-bit)

Input spatial: 224×224 (stride=2 → output 112×112)

```
MACs  = 64 × 112 × 112 × 32 × 9 = 231,211,008
FLOPs = 462,422,016

Bytes = (64×32×9 + 32×224×224 + 64×112×112) × 4
      = 9,707,520 B

AI = 462,422,016 / 9,707,520 = 47.6 FLOP/byte
```

### conv3 — Chiplet (SW: FP32, HW: 1-bit)

Input spatial: 112×112 (stride=2 → output 56×56)

```
MACs  = 128 × 56 × 56 × 64 × 9 = 231,211,008
FLOPs = 462,422,016

Bytes = (128×64×9 + 64×112×112 + 128×56×56) × 4
      = 5,111,808 B

AI = 462,422,016 / 5,111,808 = 90.5 FLOP/byte
```

### conv4 — Chiplet (SW: FP32, HW: 1-bit)

Input spatial: 56×56 (stride=2 → output 28×28)

```
MACs  = 256 × 28 × 28 × 128 × 9 = 231,211,008
FLOPs = 462,422,016

Bytes = (256×128×9 + 128×56×56 + 256×28×28) × 4
      = 3,588,096 B

AI = 462,422,016 / 3,588,096 = 128.9 FLOP/byte
```

### Whole-model (SW float32 baseline)

```
Total FLOPs = 86,704,128 + 462,422,016 × 3 = 1,473,970,176
Total Bytes = 7,028,096 + 9,707,520 + 5,111,808 + 3,588,096 = 25,435,520

AI_sw = 1,473,970,176 / 25,435,520 = 57.9 FLOP/byte
```

**Whole-model SW AI: 57.9 FLOP/byte** (up from 46.3 FLOP/byte for the 3-layer baseline)

---

## 3. Hardware Chiplet Arithmetic Intensity (1-bit BNN)

On the chiplet, weights and activations are stored and transferred at 1-bit. The host sends conv1 output at 8-bit over AXI4-Stream.

### Memory traffic

| Item | Size |
|---|---|
| AXI input (8-bit, conv1 output) | 32 × 224 × 224 × 1 B = 1,605,632 B |
| conv2 weights (1-bit) | 64×32×9 ÷ 8 = 2,304 B |
| conv2 output (1-bit) | 64×112×112 ÷ 8 = 100,352 B |
| conv3 weights (1-bit) | 128×64×9 ÷ 8 = 9,216 B |
| conv3 output (1-bit) | 128×56×56 ÷ 8 = 50,176 B |
| conv4 weights (1-bit) | 256×128×9 ÷ 8 = 36,864 B |
| conv4 output (1-bit) | 256×28×28 ÷ 8 = 25,088 B |
| **Total** | **1,829,632 B** |

### FLOPs (XNOR+Popcount, counted as 1 op per MAC)

```
Conv2-4 MACs = 231,211,008 × 3 = 693,633,024
```

### Hardware AI

```
AI_hw = 693,633,024 / 1,829,632 = 379.1 FLOP/byte
```

**Hardware chiplet AI: 379.1 FLOP/byte** — well into the compute-bound regime of both M1 and M5.

---

## 4. Comparison Table

| Configuration | AI (FLOP/byte) | Regime (M1 ridge = 38.1) |
|---|---|---|
| Old SW baseline (3-layer, conv1 dominant) | 12.3 | Memory-bound |
| Old SW baseline (3-layer, whole model) | 46.3 | Marginally compute-bound |
| **New SW baseline (4-layer hybrid, whole model)** | **57.9** | Compute-bound |
| **Hardware chiplet (1-bit Conv2-4)** | **379.1** | Deeply compute-bound |

The hybrid partition moves the operating point ~30× to the right of the M1 ridge, fully eliminating the memory wall for the binary layers.

---

## 5. Measured Attained Performance

| Configuration | Platform | Measured |
|---|---|---|
| Old SW (3-layer, CPU) | M1 | 83.0 GFLOP/s |
| New SW (4-layer hybrid, CPU) | M5 | 116.3 GFLOP/s |
