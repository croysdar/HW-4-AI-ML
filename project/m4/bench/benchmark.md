# M4 Benchmark Report — BNN Accelerator

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

## 1. Platform and Configurations Compared

| Config               | Platform                                       | Clock   | Weight Memory                        | Status                                         |
| -------------------- | ---------------------------------------------- | ------- | ------------------------------------ | ---------------------------------------------- |
| **SW Baseline**      | Apple M5 CPU (float32 PyTorch, bnn_serengeti2) | N/A     | N/A                                  | Done (serengeti2_profile.py)                   |
| **Reg-file (WD=64)** | Sky130A HD, OpenLane 2.3.10                    | 100 MHz | 64×256-bit reg file                  | P&R PASS                                       |
| **1-SRAM macro**     | Sky130A HD                                     | 20 MHz  | 1× sky130_sram_1kbyte_1rw1r_32x256_8 | P&R PASS                                       |
| **4-SRAM macro**     | Sky130A HD                                     | 40 MHz  | 4× sky130_sram_1kbyte_1rw1r_32x256_8 | P&R complete (DRC bypassed) — **final design** |
| **8-SRAM macro**     | Sky130A HD                                     | 40 MHz  | 8× sky130_sram_1kbyte_1rw1r_32x256_8 | P&R PASS (DRC bypassed)                        |

Raw numbers: [benchmark_data.csv](benchmark_data.csv)

Note on binary operations: the accelerator performs XNOR+popcount (1-bit equivalent MACs),
not float32 MACs. The M5 software baseline runs the same BNN network in PyTorch but emulates
binary operations via float32 tensors — this is why the CPU appears slow relative to
its peak FLOP/s. The comparison is valid for the target task (BNN inference), but
FLOP/s numbers are architectural equivalents, not standard IEEE floating-point metrics.

> **Baseline evolution:** The original M1 placeholder (`project/m1/sw_baseline.md`) used
> a 3-layer model and measured 82 FPS at 12.19 ms/frame. The algorithm was extended to
> 4 binary layers for accuracy gain; the final trained model (`bnn_serengeti2.pth`) was
> re-profiled on an Apple M5 (`project/serengeti2_profile.py`), yielding 151.5 FPS at
> 6.6 ms/frame. All M4 figures use the M5/4-layer baseline.

---

## 2. Throughput

### Software Baseline (Apple M5, bnn_serengeti2)

| Metric                           | Value                             |
| -------------------------------- | --------------------------------- |
| Mean latency (single image)      | 6.6 ms                            |
| Throughput                       | 151.5 images/sec                  |
| Attained compute                 | ~223 GFLOP/s (float32 equivalent) |
| Binary layers (conv2–4) fraction | ~78% of runtime (519 ms / 663 ms) |
| Arithmetic intensity (conv2)     | ~47.64 FLOP/byte                  |

_Source: `project/serengeti2_profile.py`, 100-run wall-clock average, bnn_serengeti2.pth (4-layer, 1.47 GFLOP)_

### Hardware Configurations

The full BNN model (bnn_serengeti2) processes layers conv2–4 on the accelerator.
Per-layer tile counts: conv2 = 12,544 spatial × 64 channels = **802,816 tiles**,
conv3 = 3,136 × 128 = **401,408 tiles**, conv4 = 784 × 256 = **200,704 tiles**.
Total: **1,404,928 tiles**.

#### Reg-file baseline (WD=64, 100 MHz)

Parallel 256-bit weight reads from the register file; no serialization overhead.
Per-tile cycles ≈ n_beats + 3 (drain). Full-frame estimate:

| Layer     | Beats/tile | Est. cycles/tile | Tiles     | Est. cycles |
| --------- | ---------- | ---------------- | --------- | ----------- |
| conv2     | 2          | ~5               | 802,816   | ~4.0M       |
| conv3     | 3          | ~6               | 401,408   | ~2.4M       |
| conv4     | 5          | ~8               | 200,704   | ~1.6M       |
| **Total** |            |                  | 1,404,928 | **~8.0M**   |

At 100 MHz: ~8.0M × 10 ns = **~80 ms → ~12.5 FPS** (analytical estimate; not simulated).

#### 1-SRAM macro (20 MHz)

32-bit SRAM reads with 8 serial chunks per beat. Per-tile cycles = n_beats×8 + 7.
Confirmed by simulation (`sram_1macro_experiment/sim/tb_timing.sv`):

| Layer     | Beats/tile | Cycles/tile | Tiles     | Cycles         |
| --------- | ---------- | ----------- | --------- | -------------- |
| conv2     | 2          | **23**      | 802,816   | 18,464,768     |
| conv3     | 3          | **31**      | 401,408   | 12,443,648     |
| conv4     | 5          | **47**      | 200,704   | 9,433,088      |
| **Total** |            |             | 1,404,928 | **40,341,504** |

At 20 MHz (50 ns/cycle): 40,341,504 × 50 ns = **2,017 ms → 0.50 FPS**

#### 4-SRAM macro (40 MHz)

2-phase SRAM reads: each 256-bit weight requires 2 cycles (phase 0 = lower 128 bits, phase 1 = upper 128 bits).
AXI stalls 1 cycle per beat during phase 1. Per-tile cycles = 2×n_beats + 6.
**Confirmed by iverilog simulation** (`sram_4macro_experiment/sim/tb_timing_4macro.sv`):

| Layer     | Beats/tile | Cycles/tile | Tiles     | Cycles         |
| --------- | ---------- | ----------- | --------- | -------------- |
| conv2     | 2          | **10**      | 802,816   | 8,028,160      |
| conv3     | 3          | **12**      | 401,408   | 4,816,896      |
| conv4     | 5          | **16**      | 200,704   | 3,211,264      |
| **Total** |            |             | 1,404,928 | **16,056,320** |

At 40 MHz (25 ns/cycle): 16,056,320 × 25 ns = **401 ms → 2.5 FPS**

Power: **11.53 mW** (post-route OpenSTA, TT 25°C 1.8V). Std-cell area: 75,802 µm². Die: 4.32 mm².

#### 8-SRAM macro (40 MHz)

Parallel 256-bit SRAM reads (8 banks, 1 cycle). Per-tile cycles = n_beats + 6.
**Confirmed by iverilog simulation** (`sram_8macro_experiment/sim/timing_sim.log`):

| Layer     | Beats/tile | Cycles/tile | Tiles     | Cycles         |
| --------- | ---------- | ----------- | --------- | -------------- |
| conv2     | 2          | **8**       | 802,816   | 6,422,528      |
| conv3     | 3          | **9**       | 401,408   | 3,612,672      |
| conv4     | 5          | **11**      | 200,704   | 2,207,744      |
| **Total** |            |             | 1,404,928 | **12,242,944** |

At 40 MHz (25 ns/cycle): 12,242,944 × 25 ns = **306.1 ms → 3.3 FPS**

### Hardware vs. Software Summary

| Metric                  | SW Baseline (M5) | 1-macro HW      | 4-macro HW      | 8-macro HW      |
| ----------------------- | ---------------- | --------------- | --------------- | --------------- |
| Frame time (BNN layers) | 6.6 ms           | 2,017 ms        | 401 ms          | 306 ms          |
| Throughput              | 151.5 FPS        | 0.50 FPS        | 2.5 FPS         | 3.3 FPS         |
| Speedup (frame time)    | 1×               | 0.003× (slower) | 0.016× (slower) | 0.022× (slower) |

The hardware is slower than the M5 CPU for these layers. The M5 achieves high
throughput through vectorized PyTorch operations, SIMD acceleration, batched
execution, and cache warmth — none of which apply to the serial tile-by-tile
hardware execution without pipelining across tiles. The hardware advantage is
**energy efficiency**, not raw throughput.

**Why hardware is slower:** The per-tile sequential execution (send beats, wait
for drain, repeat) means only one tile is in flight at a time. Pipelining the
drain across tiles — starting the next tile's beats while the current tile's
drain cycles run — would reduce effective cycles/tile from `n_beats + 6` to
approximately `max(n_beats, 6)`. For conv2 (2 beats), this improvement would
be 8 → 6 cycles; for conv4 (5 beats), 11 → 6 cycles. Full pipelining is
identified as the primary path to ≥30 FPS.

---

## 3. Power

| Config                    | Power                  | Source                               |
| ------------------------- | ---------------------- | ------------------------------------ |
| SW Baseline (M1 SoC)      | ~10,000 mW (estimated) | Published review (Anandtech 2020)    |
| Reg-file (WD=64, 100 MHz) | 215.3 mW               | `synth/power_report.txt`             |
| 1-SRAM macro (20 MHz)     | **2.91 mW**            | `sram_1macro_experiment/` post-route |
| 4-SRAM macro (40 MHz)     | **11.53 mW**          | `sram_4macro_experiment/` post-route |
| 8-SRAM macro (40 MHz)     | **17.78 mW**           | `sram_8macro_experiment/` post-route |

All hardware power figures: OpenSTA post-route, nominal corner (TT 25°C 1.8V).

### 1-SRAM Power Breakdown

| Component            | Power       | Share |
| -------------------- | ----------- | ----- |
| Sequential (947 FFs) | 0.82 mW     | 28%   |
| Clock distribution   | 1.02 mW     | 35%   |
| SRAM macro           | 0.79 mW     | 27%   |
| Combinational        | 0.29 mW     | 10%   |
| **Total**            | **2.91 mW** |       |

### 8-SRAM Power Breakdown

| Component                 | Power        | Share |
| ------------------------- | ------------ | ----- |
| Internal (macros + logic) | 15.53 mW     | 87%   |
| Switching                 | 2.10 mW      | 12%   |
| Leakage                   | 0.15 mW      | 1%    |
| **Total**                 | **17.78 mW** |       |

---

## 4. Energy per Frame

| Config             | Power      | Frame time | Energy/frame |
| ------------------ | ---------- | ---------- | ------------ |
| M1 CPU (estimated) | ~10,000 mW | 12.19 ms   | ~122,000 µJ  |
| 1-macro HW         | 2.91 mW    | 2,017 ms   | 5,869 µJ     |
| 4-macro HW         | 11.53 mW  | 401 ms     | **4,620 µJ** |
| 8-macro HW         | 17.78 mW   | 306.1 ms   | **5,442 µJ** |

**~22× better energy/frame** than M1 despite lower throughput, because the hardware
draws ~3,500× less power and runs only the binary layers (not full model).

---

## 5. Area

| Config           | Std-cell area                                  | Die area | Utilization |
| ---------------- | ---------------------------------------------- | -------- | ----------- |
| Reg-file (WD=64) | 1,044,000 µm²                                  | 2.56 mm² | 40.8%       |
| 1-SRAM macro     | 306,714 µm²                                    | 4.0 mm²  | 3.2%        |
| 4-SRAM macro     | 75,802 µm² (stdcell) + ~762,800 µm² (macros)   | 4.32 mm² | ~19.8%      |
| 8-SRAM macro     | 124,129 µm² (stdcell) + 1,525,700 µm² (macros) | 5.76 mm² | ~28.6%      |

---

## 6. Roofline Analysis

See [figures/roofline_final.png](figures/roofline_final.png) for the annotated plot.

### Key Operating Points

| System                              | Arithmetic Intensity (FLOP/byte) | Attained Performance | Region        |
| ----------------------------------- | -------------------------------- | -------------------- | ------------- |
| Apple M5 CPU (4-layer model)        | ~47.64 FLOP/byte                 | ~223 GFLOP/s         | Near ridge    |
| BNN chiplet (40 MHz, 4-macro final) | ~1,975 FLOP/byte                 | ~1.74 GOPS equiv.    | Compute-bound |

**Arithmetic intensity calculation (hardware):**

- AXI payload per frame (binary-packed activations only): ~0.35 MB
  - conv2 input: 32ch × 224² = 1,605,632 bits = 200,704 B
  - conv3 input: 64ch × 112² = 802,816 bits = 100,352 B
  - conv4 input: 128ch × 56² = 401,408 bits = 50,176 B
- Operations per frame: ~694 MOp (Conv2–4 XNOR-popcount, 231M each)
- AI = 694×10⁶ / 351,232 ≈ **~1,975 FLOP/byte**

At ~1,975 FLOP/byte, the hardware is strongly compute-bound. The activation-streaming
dataflow (weights on-chip in SRAM, binary-packed activations streaming via AXI) is what
drives arithmetic intensity so high — the AXI bus carries only 1-bit-packed activation
data, making each transferred byte correspond to nearly 2,000 on-chip operations.

**Attained performance at 2.5 FPS (final 4-macro design):**
2.5 FPS × 694 MOp/frame = **~1.74 GOPS** (XNOR equivalent)

For reference, the 8-macro experiment attains 3.3 FPS × 694 MOp/frame = ~2.29 GOPS.

---

## 7. Design Efficiency

### Cross-Configuration Comparison

| Metric         | 1-macro      | 4-macro       | 8-macro       |
| -------------- | ------------ | ------------- | ------------- |
| Clock          | 20 MHz       | 40 MHz        | 40 MHz        |
| Cycles/frame   | 40,341,504   | 16,056,320    | 12,242,944    |
| Frame time     | 2,017 ms     | 401 ms        | 306 ms        |
| **Throughput** | **0.50 FPS** | **2.5 FPS**   | **3.3 FPS**   |
| Power          | 2.91 mW      | **11.53 mW** | 17.78 mW      |
| Energy/frame   | 5,869 µJ     | **4,620 µJ**  | 5,442 µJ      |
| Die area       | 4.0 mm²      | 4.32 mm²      | 5.76 mm²      |
| Routing DRC    | 0            | 5 (bypassed)  | 12 (bypassed) |
| KLayout DRC    | 0            | **0**         | 8 (bypassed)  |
| LVS errors     | 0            | 7 (bypassed)  | 15 (bypassed) |

The 4-macro design sits between 1-macro and 8-macro: 5× faster than 1-macro at the same
40 MHz clock, 1.3× slower than 8-macro with cleaner DRC (KLayout 0 vs 8, routing 5 vs 12).
The 2-phase SRAM read (1 stall cycle per beat) reduces effective throughput vs 8-macro:
cycles/tile = 2×n_beats + 6 vs n_beats + 6 for 8-macro.

The 8-macro design achieves 6.6× throughput improvement vs 1-macro at 6.1× more power —
energy/frame is approximately equal. The throughput improvement is 6.6× (not 16× as
8× SRAM banks × 2× clock would suggest) because the fixed 6-cycle drain overhead per
tile is not reduced by the memory bandwidth increase.

---

_Data sources: `synth/power_report.txt`, `synth/area_report.txt`, `synth/timing_report.txt`,
`sram_1macro_experiment/` P&R reports, `sram_8macro_experiment/sim/timing_sim.log`,
`project/serengeti2_profile.py` (SW baseline — M5, bnn_serengeti2, 4-layer)_
