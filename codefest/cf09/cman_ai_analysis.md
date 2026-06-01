# CF09 CMAN -- Arithmetic Intensity of the BNN Accelerator Kernel

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

## Task 1 -- Dominant Kernel: Dimensions and Data Type

**Kernel:** 1-bit XNOR-Popcount dot product (the inner loop of binary convolution)

The hardware accelerator (`compute_core`) implements the dominant kernel for
BinarizeConv2d layers conv2 and conv3. Each kernel invocation computes one
256-wide binary dot product:

```
xnor_bits = ~(act_in XOR weight_in)           // 256-bit bitwise XNOR
popcount  = popcount(xnor_bits)               // Hamming weight, 0-256
dot_val   = 2*popcount - 256                  // maps to [-256, +256]
accum_out += dot_val                          // 32-bit signed accumulator
```

**Dimensions at operating point (one tile invocation):**

| Parameter           | Symbol     | Value                           | Notes                                                                                    |
| ------------------- | ---------- | ------------------------------- | ---------------------------------------------------------------------------------------- |
| Vector width        | N          | 256 bits                        | `VECTOR_WIDTH` parameter in RTL                                                          |
| Input channel depth | Cin        | 32 or 64                        | conv2: 32, conv3: 64 -- zero-padded to 256 to match the accelerator's fixed vector width |
| Kernel window       | KxK        | 3x3 = 9                         | spatial receptive field (each output pixel sees a 3x3 patch of the input)                |
| Output channels     | Cout       | 64 (conv2) / 128 (conv3)        |                                                                                          |
| Output spatial      | Hout\*Wout | 112x112 (conv2) / 56x56 (conv3) | stride-2 halves dims each layer: 224->112->56                                            |

**Data types:**

| Signal    | Width | Type                                     |
| --------- | ----- | ---------------------------------------- |
| act_in    | 256 b | packed binary (1-bit per activation)     |
| weight_in | 256 b | packed binary (1-bit per weight)         |
| accum_out | 32 b  | signed integer (accumulated dot product) |

The accelerator processes one 256-bit tile per clock cycle. For conv2, Cin=32 so
only 32 of the 256 bits are real data -- the rest are zero-padded. For conv3,
Cin=64, same deal but less wasteful. [1]

**Operating point:** conv3 (64->128, 56x56 output) -- bigger output volume, used for
all calculations below.

---

## Task 2 -- FLOP Count

**Method:** Count XNOR operations + popcount additions for one full conv3 forward pass.

The standard formula for binary convolution FLOPs (counting each 1-bit XNOR as 1 OP
and each addition as 1 OP, consistent with MACs\*2 convention):

```
FLOPs = 2 * Cin * K * K * Hout * Wout * Cout
      = 2 * 64  * 3 * 3 * 56  * 56   * 128
```

Step-by-step:

```
K*K          = 9
Cin * K^2    = 64 * 9 = 576
Hout * Wout  = 56 * 56 = 3,136
FLOPs/output = 2 * 576 = 1,152
Total FLOPs  = 1,152 * 3,136 * 128
             = 1,152 * 401,408
             = 462,422,016  ~462 M FLOPs
```

(Cross-check with M1 torchinfo: conv3 reported 231,211,008 MACs \* 2 = 462,422,016 FLOPs. confirmed)

**Total FLOPs for one conv3 invocation: 462,422,016 FLOPs (~462 MFLOP)**

At the hardware level each `compute_core` invocation processes one 256-wide dot
product = 256 XNOR ops + 255 additions = **511 ops ~512 FLOPs per cycle**.
At 300 MHz: **153.6 GFLOP/s peak compute**.

---

## Task 3 -- Byte Transfer Counts (Two Bounds)

### Reuse Pattern

This kernel is a standard **GEMM-style (weight-stationary) binary matrix multiply**.
The weight tensor for one conv3 layer is 128 \* 576 bits = 73,728 bits = **9,216 bytes**.
The activation tensor (input feature map) is 64 \* 58 \* 58 \* 1 bit (with padding) ~
**27,556 bytes**. Both fit entirely in the 1,512-word \* 256-bit on-chip weight register
file (386 KB capacity).

**Reuse pattern: weight-stationary. Weights are loaded once off-chip per layer;
activations stream in tile by tile.**

---

### Bound A: No Data Reuse (Lower Bound on AI)

In this case every weight bit and every activation bit is re-read from off-chip memory for every
output element -- no caching whatsoever.

No reuse means every multiply re-reads both operands from off-chip:

```
Bytes_weights_no_reuse = Cin * K^2 * Cout * Hout * Wout / 8
                       = 576 * 128 * 3,136 / 8
                       = 28,901,376 bytes

Bytes_acts_no_reuse    = Cin * K^2 * Hout * Wout * Cout / 8
                       = 576 * 3,136 * 128 / 8
                       = 28,901,376 bytes  (same by symmetry)

Bytes_output           = Cout * Hout * Wout * 32 bits / 8
                       = 128 * 3,136 * 4 = 1,605,632 bytes

Total_bytes_no_reuse   = 28,901,376 + 28,901,376 + 1,605,632
                       = 59,408,384 bytes  ~59.4 MB
```

**AI (lower bound, no reuse):**

```
AI_lower = FLOPs / Bytes_no_reuse
         = 462,422,016 / 59,408,384
         = 7.78 FLOP/byte  ~7.8 FLOP/byte
```

---

### Bound B: Perfect On-Chip Weight Reuse (Upper Bound on AI)

Weights are loaded once from off-chip; activations stream once (input featuremap
read once per spatial position); outputs written once.

```
Bytes_weights_reuse = Cin * K^2 * Cout / 8
                    = 576 * 128 / 8
                    = 9,216 bytes

Bytes_acts_reuse    = Cin * (Hin + pad) * (Win + pad) / 8
                    ~ 64 * 58 * 58 / 8
                    = 214,336 / 8 = 26,792 bytes
                    (approximate; includes 1-px zero-pad on each side for 3x3 conv)

Bytes_output_reuse  = Cout * Hout * Wout * 32 / 8
                    = 128 * 3,136 * 4 = 1,605,632 bytes

Total_bytes_reuse   = 9,216 + 26,792 + 1,605,632
                    = 1,641,640 bytes  ~1.64 MB
```

**AI (upper bound, full weight reuse):**

```
AI_upper = FLOPs / Bytes_reuse
         = 462,422,016 / 1,641,640
         = 281.7 FLOP/byte  ~282 FLOP/byte
```

---

## Task 4 -- Arithmetic Intensity Summary and Roofline

| Bound                     | Bytes transferred | AI (FLOP/byte) |
| ------------------------- | ----------------- | -------------- |
| Lower (no reuse)          | 59,408,384        | **7.8**        |
| Upper (full weight reuse) | 1,641,640         | **282**        |

### Sky130 HD Nominal Platform Figures

| Parameter                                            | Value          | Source                   |
| ---------------------------------------------------- | -------------- | ------------------------ |
| Peak compute (300 MHz, 256-wide XNOR-Popcount)       | **153.6 GOPS** | 512 ops/cycle \* 300 MHz |
| Off-chip bandwidth (AXI4-Stream 256b @ 300 MHz)      | **9.6 GB/s**   | interface_selection.md   |
| On-chip SRAM bandwidth (1 read port, 256b @ 300 MHz) | **9.6 GB/s**   | same clock domain        |

**Ridge point:**

```
Ridge_AI = Peak_compute / Peak_BW
         = 153.6 GOPS / 9.6 GB/s
         = 16 FLOP/byte
```

### Attainable Performance at Each AI Bound

```
Perf(AI_lower = 7.8)  = min(153.6, 7.8 * 9.6)  = min(153.6, 74.9)  = 74.9 GOPS
Perf(AI_upper = 282)  = min(153.6, 282 * 9.6)   = min(153.6, 2,707) = 153.6 GOPS
```

- At **AI = 7.8** (no reuse): memory-bound, attainable = **74.9 GOPS**
- At **AI = 282** (full reuse): compute-bound, attainable = **153.6 GOPS** (at ceiling)

**Roofline sketch:** see `codefest/cf09/cman_roofline_sketch.png`

---

## Task 5 -- Bottleneck Identification and Improvement

### Current Bottleneck

The design is limited by **on-chip memory bandwidth** -- specifically the single read
port of the 256-bit weight register file (or SRAM in M4), which delivers exactly one
256-bit weight vector per cycle. This matches the AXI4-Stream input bandwidth, so the
system is balanced at the interface but the on-chip weight memory is the single-port
bottleneck that prevents issuing multiple MAC tiles per cycle.

At AI = 7.8 (no reuse / cold weights), the kernel is below the ridge point and
memory-bandwidth-limited. In normal operation weights stay resident in SRAM across
the full layer, so AI = 282 and the kernel is compute-bound at the 153.6 GOPS ceiling.

The actual bottleneck at that ceiling is just the single `compute_core` unit -- one
dot product per cycle, no way to go faster without more parallelism.

### Highest-Leverage Improvement

**Instantiate N parallel `compute_core` units** (vector parallelism) to process
N output channels simultaneously in the same clock cycle. With N=8 cores:

```
Peak compute (*8 cores) = 8 * 153.6 = 1,228.8 GOPS
```

That gets close to the 1,200 GFLOP/s interface target from M1. The RTL change is
straightforward (8 accumulator arrays + output mux); the hard part is making the
weight SRAM supply 8 vectors/cycle without becoming the new bottleneck.

---

[1] For conv2 (Cin=32), 224 of the 256 input bits are zero-padding. XNORing two zeros
gives 1, so those positions add a constant +224 bias to every popcount -- but that
cancels out in `2*popcount - 256`, so correctness is fine. The real issue is that
conv2 is only using 12.5% of the hardware each cycle (conv3 is 25%). The numbers in
this analysis don't correct for that -- they reflect what the hardware actually does.
