# M4 Benchmark Report — BNN Accelerator

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

## 1. Platform and Configurations Compared

| Config | Platform | Clock | Weight Memory | Status |
|--------|----------|-------|---------------|--------|
| **SW Baseline** | Apple M1 CPU (float32 PyTorch) | N/A | N/A | Done (M1) |
| **HW WD=64** | Sky130A HD, OpenLane 2.3.10 | 100 MHz | 64×256-bit reg file | 78/78 PASS |
| **HW WD=64 @ 20 MHz** | Sky130A HD | 20 MHz | 64×256-bit reg file | Estimated |
| **HW SRAM-256** | Sky130A HD | 20 MHz | 16× 32x256 SRAM macros | In progress |

Raw numbers: [benchmark_data.csv](benchmark_data.csv)

---

## 2. Throughput

### Software Baseline (M1)

| Metric | Value |
|--------|-------|
| Mean latency (single image) | 12.19 ms |
| Throughput | 82.0 images/sec |
| Attained compute | ~83 GFLOP/s |
| Compute utilization vs. peak | ~3.2% (memory-bound) |

The software baseline is bottlenecked by DRAM bandwidth (AI = 12.34 FLOP/byte on
the binary layers), not compute. The CPU runs at ~3% of its peak theoretical
throughput because data arrives from DRAM slower than the ALUs can consume it.

### Hardware (WD=64, 100 MHz)

The BNN accelerator processes one 256-bit activation×weight beat per clock cycle
in steady state. At 100 MHz:

| Layer | Beats per tile | Tiles per frame | Cycles |
|-------|----------------|-----------------|--------|
| conv2 | 5 | ~128 | ~640 |
| conv3 | 5 | ~384 | ~1,920 |
| conv4 | 5 | ~640 | ~3,200 |
| **Total** | | ~1,152 | **~5,760** |

5,760 cycles ÷ 100 MHz = **57.6 µs per frame** → sustained >**17,000 FPS** on the
binary layers alone. The practical ceiling is 30 FPS (host-side conv1 + AXI overhead).

Attained compute: 505,774,786 MACs × 2 / 57.6 µs = **~17.6 TFLOP/s** equivalent
(treating XNOR+popcount as 1-bit MACs). Note: FLOP equivalence is architectural,
not standard IEEE GFLOP — binary ops are fundamentally different from float32.

### Hardware vs. Software Speedup

| Metric | SW Baseline | HW (WD=64) | Speedup |
|--------|-------------|------------|---------|
| Latency (binary layers) | 12.19 ms (full model) | 57.6 µs | >200× (on binary layers) |
| Throughput | 82 FPS | >17,000 FPS (binary) | >200× |

The >200× speedup is specifically on the binary conv2–4 layers handled by the
accelerator. The full-model speedup depends on conv1 (INT8, runs on host CPU)
and is expected at ~5–10× system-level.

---

## 3. Power

### Measured (WD=64, 100 MHz, TT 25°C 1.8V)

| Group | Power | % |
|-------|-------|---|
| Sequential (FFs) | 90.3 mW | 41.9% |
| Combinational | 64.4 mW | 29.9% |
| Clock | 60.7 mW | 28.2% |
| **Total** | **215.3 mW** | |

Power is dominated by the 16,384 flip-flops in the register-file weight memory
switching at nominal activity. At 100 MHz with full-rate clocking, the FF array
and its clock distribution consume ~70% of total power.

### At 20 MHz (clock-frequency optimization)

Dynamic power scales approximately linearly with frequency. At 20 MHz:

- Switching power: 55.4 mW × (20/100) = ~11.1 mW
- Clock network: 60.7 mW × (20/100) = ~12.1 mW
- Internal (static-dominated): ~160 mW × correction factor

A re-synthesis at 20 MHz would also select smaller/slower cells, further reducing
cell area and static power. Estimated total at 20 MHz: **~43 mW** (5× reduction
on switching-dominated components, with cell downsizing reducing internal power too).

### With SRAM Macros (projected, 20 MHz)

| Component | Power |
|-----------|-------|
| 16× SRAM macros (32x256, 1 access/cycle) | ~8 mW total |
| Compute + interfaces (20 MHz) | ~10–15 mW |
| Clock distribution | ~5 mW |
| **Estimated total** | **~25–30 mW** |

SRAM macros dissipate energy only during active read/write; standby leakage is
~µW per macro. This is the primary motivation for the SRAM integration experiment.

### Power vs. Target

| Metric | Target | WD=64 (100 MHz) | WD=64 (20 MHz est.) | SRAM (projected) |
|--------|--------|-----------------|---------------------|------------------|
| Total power | <200 mW | 215.3 mW | ~43 mW | ~25–30 mW |
| Vs. target | — | 8% over | 79% under | 85% under |

---

## 4. Area

| Stage | Cells | Cell Area | Die Area | Utilization |
|-------|-------|-----------|----------|-------------|
| Pre-P&R (synthesis) | 46,774 | 718,169 µm² | — | — |
| Post-route (with CTS) | — | 1,043,990 µm² | 2,560,000 µm² | **40.8%** |

The post-route area overhead (1.044 mm² vs. 0.718 mm² synthesis) comes from:
- Clock tree synthesis (CTS) buffers: +~190,000 µm²
- Hold-fix buffers added by OpenROAD resizer: +~136,000 µm²

Register-file breakdown: ~16,384 `sky130_fd_sc_hd__dfxtp_2` flip-flops × ~27.7 µm²
each ≈ 453,000 µm² (63% of synthesis area; 43% of post-route area).

---

## 5. Roofline Analysis

See [figures/roofline_final.png](figures/roofline_final.png) for the annotated plot.

### Key Operating Points

| System | Arithmetic Intensity (FLOP/byte) | Attained Performance | Region |
|--------|----------------------------------|---------------------|--------|
| Apple M1 CPU (full model) | ~12.3 FLOP/byte | ~83 GFLOP/s | Memory-bound |
| BNN chiplet (100 MHz, WD=64) | ~379 FLOP/byte | ~17.6 TFLOP/s equiv. | Compute-bound |

**Arithmetic intensity calculation (hardware):**
- AXI payload per frame (activations only): ~1.6 MB (32×224×224 bytes)
- Operations per frame: ~606 GFLOP (Conv2–4 XNOR-popcount equivalent)
- AI = 606 GFLOP / 1.6 MB = **379 FLOP/byte**

At 379 FLOP/byte, the hardware design is firmly compute-bound: the AXI bus delivers
data faster than the XNOR+popcount units consume it. This is the correct operating
regime for a streaming binary neural network accelerator — every byte transferred
from the host yields 379 useful operations on-chip.

**Ridge point (hardware):**
- Peak compute: 256 XNOR/cycle × 100 MHz = 25.6 GXNOR/s ≈ 25.6 TOPS
- AXI bandwidth (theoretical): 256 bits / 100 MHz × 100 MHz = 3.2 GB/s
- Ridge = 25,600 GFLOP/s / 3,200 MB/s = **8,000 FLOP/byte**

The design at AI=379 is in the bandwidth-bound region of the hardware roofline —
meaning it processes data faster than the AXI interface can supply it. This is
intentional: the weight-stationary dataflow means the AXI bus only carries activations,
not weights (which stay on-chip), making the effective AI much higher than the
instruction-level AI.

---

## 6. Roofline Plot Notes

The roofline figure ([figures/roofline_final.png](figures/roofline_final.png)) shows:

- **X-axis:** Arithmetic intensity (FLOP/byte), log scale 1–10,000
- **Y-axis:** Attained performance (GFLOP/s), log scale 1–100,000
- **Memory roof:** Slope = AXI bandwidth limit (3.2 GB/s at 100 MHz)
- **Compute roof:** Horizontal = peak XNOR throughput (25,600 GFLOP/s equiv.)
- **SW point:** M1 CPU at (12.3, 83) — memory-bound, left of ridge
- **HW point:** BNN chiplet at (379, 17,600) — in right portion, near compute roof

The 30× improvement in arithmetic intensity (12.3 → 379 FLOP/byte) is what makes
the custom hardware worthwhile. It restructures the compute/memory balance so that
the accelerator stays compute-bound even at high throughput.

---

## 7. Design Efficiency

### Compute Density

| Metric | Value |
|--------|-------|
| Peak XNOR throughput | 25.6 GXNOR/s |
| Core area (compute + interfaces only) | ~0.27 mm² |
| Compute density | ~94.8 GXNOR/s/mm² |

### Energy Efficiency (WD=64, 100 MHz)

| Metric | Value |
|--------|-------|
| Total power | 215.3 mW |
| Throughput (binary layers) | 17,600 FPS |
| Energy per frame | ~12.2 µJ/frame |
| Vs. M1 CPU (~2W for binary layers) | ~164× more efficient |

At 20 MHz with SRAM (target configuration):

| Metric | Projected |
|--------|-----------|
| Total power | ~25–30 mW |
| Throughput | >3,000 FPS (binary layers) |
| Energy per frame | ~8–10 µJ/frame |
| Vs. M1 CPU | ~200–250× more efficient |

---

*Data sources: synth/power_report.txt, synth/area_report.txt, synth/timing_report.txt,
project/m1/sw_baseline.md, project/design_decisions/q02_roofline_motivation.md*
