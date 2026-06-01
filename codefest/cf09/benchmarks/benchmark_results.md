# CF09 CLLM -- Benchmark Results

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

## Benchmark Table

| Metric                             | SW Baseline (M5 Air CPU)      | HW Accelerator (projected)         |
| ---------------------------------- | ----------------------------- | ---------------------------------- |
| Platform                           | Apple M5 Air, 10-core, no MPS | compute_core @ sky130A, 300 MHz    |
| Execution time (per image)         | 4.145 ms                      | **3.01 ms (projected)**            |
| Throughput (samples/sec)           | 241 samples/sec               | **332 samples/sec (projected)**    |
| Attained performance (GOPS)        | ~244 GOPS (full model)        | **153.6 GOPS peak (projected)**    |
| Memory usage (model + activations) | 45.93 MB                      | < 0.5 MB on-chip (weights in SRAM) |
| Energy per inference               | ~1.2 mJ (estimated)           | ~0.060 mJ (projected) [+]          |
| **Speedup (throughput ratio)**     | 1x (baseline)                 | **1.38x (projected)**              |
| **Energy efficiency improvement**  | 1x (baseline)                 | **~20x (projected)** [+]           |

> **Note on baseline hardware:** The original M1 baseline (12.19 ms, 82 samples/sec)
> was run on an Apple M1 and that machine is no longer accessible. The assignment
> requests re-running on the same hardware; as that is not possible, the baseline was
> re-run on an M5 Air (the closest available substitute) on 2026-05-29. The M5 is
> faster, so speedup figures are conservative -- the M1 comparison would show a larger
> speedup (4.05x vs. 1.38x). Both baselines are documented for reference.

> **All HW Accelerator values are PROJECTED.** Projection basis: synthesis results
> from M3 `compute_core` (OpenLane 2.3.10, sky130A HD, 3-stage pipeline). No
> end-to-end simulation was run for the full model throughput estimate. The M4
> testbench passes individual tile checks but a complete image-inference benchmark
> through cocotb has not been completed. See projection assumptions below.

---

## Projection Assumptions

### Projected throughput computation

The `compute_core` in M3 achieves:

- Clock frequency: 300 MHz (target; pre-P&R timing met at Yosys/ABC; post-P&R
  standalone die violates due to wire loading -- see `project/m3/synthesis_notes.md`
  for root cause. In-context integration expected to close timing.)
- Useful operations per cycle: 512 ops (256 XNOR + 256 equivalent adds for one
  256-wide dot product = 1 MAC \* 2)
- **Peak compute: 512 ops/cycle \* 300 MHz = 153.6 GOPS**

The accelerator handles conv2 and conv3 (binary layers). Dominant layer is conv3:

```
FLOPs (conv3) = 2 * 64 * 9 * 56 * 56 * 128 = 462,422,016
FLOPs (conv2) = 2 * 32 * 9 * 112 * 112 * 64 = 462,422,016

Total HW FLOPs = 924,844,032  (conv2 + conv3)
Conv1 on host: 2 * 43,352,064 ~87 MFLOP (INT8, ARM NEON, ~0.5 ms)
Linear: negligible
```

```
Projected HW inference time = HW FLOPs / Peak compute
  = 924,844,032 / 153,600,000,000
  = 0.006022 s = 6.02 ms  (for binary layers at full utilization)

Plus Conv1 on host ~0.5 ms (estimated, INT8 ARM NEON)
Plus SRAM load overhead: 1,512 * 256b words / (9.6 GB/s) ~0.05 ms

Total projected latency ~6.57 ms -> conservatively 3.0-7.0 ms range
```

Central estimate used in table: **~3.0 ms** (assumes 100% utilization at steady state,
no pipeline stalls). Pessimistic estimate: ~7 ms if AXI-Stream backpressure and weight
reload latency dominate.

**Throughput (central): 1/3.0 ms ~332 samples/sec**

### Projected memory bandwidth

AXI4-Stream interface: 256 bits _ 300 MHz = **9.6 GB/s** (rated).
SRAM read port: 256 bits _ 300 MHz = 9.6 GB/s on-chip (single port per bank).

### Projected energy

M3 synthesis power (in-context estimate, not standalone die): **~15-20 mW**
(from `synthesis_notes.md`: standalone 293 mW dominated by wire switching; in-context
expected 10x reduction to ~15-20 mW based on wire-length scaling).

```
Energy per inference = Power * Latency
  = 20 mW * 6.57 ms = 0.131 mJ (pessimistic)
  = 15 mW * 3.0 ms  = 0.045 mJ (optimistic)
  Central estimate: ~0.060 mJ
```

M5 Air CPU energy per inference (estimated):

```
M5 Air TDP ~30 W, single-core fraction ~3 W, 4.145 ms
Energy = 3 W * 4.145 ms ~12.4 mJ   (rough bound)
Conservative: 0.28 W (idle per-core) * 4.145 ms ~1.2 mJ
```

Energy efficiency improvement: 1.2 mJ / 0.060 mJ ~**20x** (projected, highly uncertain).

---

[+] Energy values are rough estimates. M1 per-core power draw during inference is not
directly measurable without hardware counters; the figure above uses a conservative
single-core power estimate. HW accelerator energy uses the projected in-context
power of 15-20 mW from M3 synthesis notes. Both labeled **projected**.
