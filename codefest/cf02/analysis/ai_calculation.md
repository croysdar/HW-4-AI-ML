# Arithmetic Intensity Calculation

ECE 510 Spring 2026 — Codefest 02 / M1 Deliverable

---

## 1. Dominant Kernel Identification

Profiler: `cProfile`, 100 inference passes, sorted by cumulative time.

```
ncalls  tottime  percall  cumtime   function
300     0.562    0.002    0.562     {built-in method torch.conv2d}
```

**The dominant kernel is `torch.conv2d`, accounting for 47.0% of total profiler runtime**
(0.562 s ÷ 1.196 s total profiled time).

All three `BinarizeConv2d` layers call `torch.conv2d`; combined they are the single largest contributor to runtime by a wide margin (next-largest: `torch.sign` at 0.132 s / 11.0%).

The layer chosen for the detailed arithmetic intensity calculation below is **conv1** — `BinarizeConv2d(C_in=3, C_out=32, K=3×3, stride=1, pad=1, input=224×224)` — because it operates on the full-resolution input and its AI is representative of the memory-access regime for the full conv stack.

---

## 2. FLOPs — Analytical Derivation

For a 2D convolution, the number of multiply-accumulate operations (MACs) per output element is `C_in × K × K`. The total MACs for the layer are:

```
MACs = C_out × H_out × W_out × C_in × K²
```

Substituting conv1 values:

| Symbol | Value | Meaning                                      |
| ------ | ----- | -------------------------------------------- |
| C_out  | 32    | output channels                              |
| H_out  | 224   | output height: (224 + 2×1 − 3) ÷ 1 + 1 = 224 |
| W_out  | 224   | output width                                 |
| C_in   | 3     | input channels (RGB)                         |
| K      | 3     | kernel spatial dimension                     |

```
MACs  = 32 × 224 × 224 × 3 × 3²
      = 32 × 224 × 224 × 3 × 9
      = 43,352,064

FLOPs = 2 × MACs            (1 MAC = 1 multiply + 1 add)
      = 2 × 43,352,064
      = 86,704,128
```

---

## 3. Bytes Transferred — DRAM, No Reuse

Assumption: every weight element, every input activation, and every output activation is read or written once from/to DRAM with zero on-chip data reuse. This is the standard conservative bound for roofline analysis.

Precision: **float32 (FP32), 4 bytes per element** — matching the software baseline on CPU.

### Weights

```
Bytes_weights = C_out × C_in × K × K × 4 B
              = 32 × 3 × 3 × 3 × 4
              = 3,456 B   (3.4 KB)
```

### Input Activations

```
Bytes_input = C_in × H_in × W_in × 4 B
            = 3 × 224 × 224 × 4
            = 602,112 B   (588.0 KB)
```

### Output Activations

```
Bytes_output = C_out × H_out × W_out × 4 B
             = 32 × 224 × 224 × 4
             = 6,422,528 B   (6,272.0 KB  ≈ 6.13 MB)
```

### Total

```
Bytes_total = Bytes_weights + Bytes_input + Bytes_output
            = 3,456 + 602,112 + 6,422,528
            = 7,028,096 B   (6.70 MB)
```

---

## 4. Arithmetic Intensity

```
AI = FLOPs / Bytes_total
   = 86,704,128 / 7,028,096
   = 12.34 FLOP/byte
```

**Arithmetic intensity of the conv1 kernel (FP32 software baseline): 12.34 FLOP/byte**

---

## 5. Roofline Interpretation

For a modern Apple M-series CPU:

- Peak FP32 throughput: ~2.6 TFLOP/s (M1) — see Apple spec
- Peak memory bandwidth: ~68.25 GB/s (M1 unified memory) — see Apple spec
- Ridge point: 2600 GFLOP/s ÷ 68.25 GB/s ≈ **38.1 FLOP/byte**

Since conv1's AI (12.34 FLOP/byte) is **left of the ridge point (38.1 FLOP/byte)**, the kernel is **memory-bandwidth bound** on this platform.

This motivates the hardware accelerator design: by switching weights from FP32 (4 bytes) to 1-bit (0.125 bytes), the weight memory traffic drops 32×, shifting the AI dramatically to the right toward or beyond the ridge point.
