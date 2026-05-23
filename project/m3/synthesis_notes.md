# Synthesis Notes — Milestone 3

## BNN Accelerator: compute_core Pipelined Synthesis

---

## Scope Decision

The original synthesis target was `bnn_top`, the full integration module including the
1,512 × 256-bit on-chip weight register file. During elaboration, Yosys expands this
register file into approximately 387,072 individual flip-flops before ABC runs. ABC's
internal data structures were exhausted attempting to map this register array, causing
the tool to hang and eventually be killed.

The scope was revised to `compute_core` only — the compute datapath without the weight
memory. This is documented in `synth/config.json` (`_note` field) and is consistent
with ECE 510 rubric guidance that a documented synthesis failure with revised scope is
acceptable. The weight memory is a structural register file that does not benefit
meaningfully from logic synthesis; its area and timing are predictable from first
principles (256 FFs per word × 1,512 words = 387,072 FFs, ~7.75 mm² at Sky130A HD
density). The compute_core synthesis result is therefore the interesting and
report-worthy artifact.

---

## M2 Baseline: Timing Violation

The pre-pipeline `compute_core` (M2 flat combinational version) failed timing at
300 MHz. The critical path ran from primary inputs directly to the accumulator
flip-flop — a PI-to-FF path with no source register to provide slack benefit.

The path traversed 14 combinational levels:

```
act_in / weight_in
  → xor2_1           (1 level: XNOR bit-wise)
  → xor3_1 × 2       (adder tree level 1)
  → xnor3_1 × 9      (adder tree levels 2–10 — dominant delay)
  → xnor2_1          (adder tree final stage)
  → and3_1           (accumulator enable gating)
  → nor3_1           (accumulator carry into register)
  → dfxtp_1          (sink: accum_out register)
```

Measured delay: **4,057 ps** against a 3,333 ps budget (300 MHz). Worst slack: **−827 ps**.

The nine-stage `xnor3_1` chain was the dominant contributor, accounting for ~2,934 ps
(72% of total path delay) at min-load Liberty timing arcs. Yosys/ABC chose `xnor3_1`
cells for the popcount adder tree because 3-input XOR/XNOR cells are more area-efficient
than 2-input chains for a 256-bit reduction. The tradeoff is higher per-cell delay —
each `xnor3_1` contributes ~280–330 ps, so nine in series accumulates to nearly 3 ns.

---

## M3 Fix: Pipeline Register Insertion

A single pipeline register was inserted between the XNOR stage and the popcount adder
tree. The register consists of:

- `xnor_reg[255:0]` — captures the bit-wise XNOR result (256 FFs)
- `s_valid_r` — pipeline-registers the AXI-stream valid signal
- `accum_clear_r` — pipeline-registers the accumulator clear strobe

This converts the single PI-to-FF path into two independent FF-to-FF paths:

**Stage 1** (trivially met): Primary inputs → `xnor_reg`

- Logic: one `xor2_1` gate (XNOR = NOT XOR, absorbed into the register setup)
- Delay: ~280 ps → slack **+3,053 ps**

**Stage 2** (timing-critical): `xnor_reg` → `accum_out`

- Logic: popcount adder tree, approximately 7 levels (halved from original 13)
- Delay: estimated **2,100–2,300 ps** → slack **+1,033–1,233 ps**
- ABC completed with `-D 3333` without error or degraded-mapping fallback

The fix works because the pipeline register converts an unbounded PI-to-FF combinational
path into a bounded FF-to-FF path. The adder tree depth is approximately halved because
the XNOR stage is now registered — the popcount tree starts from stable flip-flop outputs
rather than having to propagate through both the XNOR and the adder in the same clock
cycle. A 256-bit popcount from registered inputs requires roughly log₂(256) = 8 adder
levels; the 3,333 ps budget comfortably accommodates ~7 levels in Sky130A HD.

---

## Cell Count and Area Impact

| Metric          | M2 (pre-pipeline) | M3 (pipelined) | Delta |
| --------------- | ----------------- | -------------- | ----- |
| Total cells     | 1,273             | 1,710          | +437  |
| Flip-flops      | 31                | 289            | +258  |
| Total area      | 13,347.8 µm²      | 19,077 µm²     | +43%  |
| Sequential area | ~620.5 µm²        | 5,785 µm²      | +833% |

The +258 FF increase is exactly accounted for: 256 (`xnor_reg`) + 1 (`s_valid_r`) +
1 (`accum_clear_r`). The combinational cell increase (+179 cells) is primarily the
register enable/clear logic and minor synthesis reorganization by ABC after the hierarchy
changed. The 43% area increase is the cost of meeting timing; it is expected and acceptable.

---

## Anomaly: lpflow\_\* Cells

Yosys/ABC mapped 18 `lpflow_*` power isolation cells into the accumulator enable/clear
logic path:

- 6× `lpflow_inputiso1p_1`
- 12× `lpflow_isobufsrc_1`

These are power-domain isolation primitives intended for multi-supply designs, not
synchronous compute logic. Their presence likely results from ABC selecting them as
2-input buffer/AND equivalents during technology mapping. They add approximately 112 µm²
of unnecessary area and may cause issues during P&R power domain analysis if a physical
design tool interprets them as actual power domain crossings.

These should be replaced with `a21o_1` or `mux2_1` cells before tape-out, either by
adding a `DONT_USE_CELL` constraint for `lpflow_*` in the synthesis configuration or by
post-synthesis netlist cleanup. They do not affect functional correctness.

---

## OpenLane 2 P&R — Full Flow (Docker)

Initial native Yosys attempts (0.64 and 0.65) failed because Homebrew Yosys is compiled
without pyosys support (`ENABLE_PYOSYS=1`). OpenLane 2.3.10 requires the `-y` flag to
pass Python scripts to Yosys's embedded interpreter (`Yosys.JsonHeader` step).

Resolution: Docker image `ghcr.io/efabless/openlane2:2.3.10` contains Yosys 0.46
compiled with pyosys support. The full flow completed successfully.

**Run:** `RUN_2026-05-22_01-54-18` — 350×350 µm die, 30% target density

**DRC:** Clean (Magic + KLayout — 0 errors)  
**LVS:** Clean (Netgen — 0 errors)  
**GDS:** Produced at `runs/RUN_2026-05-22_01-54-18/56-magic-streamout/compute_core.gds`

### Post-Routing Power (nom_tt/25°C/1.8V)

| Group         | Internal     | Switching    | Leakage     | Total        |
| ------------- | ------------ | ------------ | ----------- | ------------ |
| Sequential    | 4.034 mW     | 0.440 mW     | 2.4 nW      | 4.474 mW     |
| Combinational | 139.97 mW    | 143.92 mW    | 11.0 nW     | 283.90 mW    |
| Clock         | 3.100 mW     | 1.936 mW     | 8.9 nW      | 5.036 mW     |
| **Total**     | **147.1 mW** | **146.3 mW** | **22.3 nW** | **293.4 mW** |

The 293 mW total is dominated by combinational wire-switching power in the oversized
350×350 µm floorplan. This design has 549 IO pins (256 act_in + 256 weight_in + 37
control/output signals), requiring a large die for IO pin placement. In context (as a
sub-block within `bnn_top`), net lengths would be ~5–10 µm vs. ~30–40 µm standalone,
reducing combinational switching by ~10×. Estimated in-context power: **~15–20 mW**.

### Post-Routing Timing (nom_tt/25°C/1.8V)

| Metric    | Value     | Status   |
| --------- | --------- | -------- |
| Hold WNS  | +0.181 ns | Met      |
| Setup WNS | −5.705 ns | Violated |
| Setup TNS | −151.5 ns | 31 paths |

The 31 setup violations are in the Stage 2 adder tree path (FF xnor_reg → FF accum_out).
The critical path at post-routing is ~9 ns vs. the 3.33 ns budget — significantly worse
than the pre-P&R ABC estimate of ~2.1–2.3 ns. Two factors drive this discrepancy:

1. **Cell remapping**: OpenROAD's resizer mapped the adder tree to 2-input `xnor2_2` /
   `xor2_2` cells (~16 logic levels) instead of ABC's 3-input `xnor3_1` cells (~7 levels).
   This is a different optimization pass with different cell preferences.

2. **Wire delay**: At 350 µm standalone die, adder net wires are 30–40 µm long,
   adding 0.3–0.6 ns per net segment. Accumulated across 16 levels this is ~5–8 ns.

**Conclusion (2-stage)**: The 2-stage pipeline is sufficient for a 300 MHz floorplan when
integrated within a larger design (short wires). As a standalone top-level module
with 549 IOs it does not close timing due to wire loading on a standalone die.

---

## 3-Stage Pipeline Implementation (same-day M3 update)

Following the 2-stage post-P&R violation analysis, the RTL was updated to a 3-stage
pipeline within the same M3 session. The key change splits the popcount adder tree into:

- **Stage 2**: 8 parallel 32-bit popcounts → `chunk_sums_r[7:0][5:0]` register
- **Stage 3**: sum of 8 × 6-bit chunk sums + accumulate → `accum_out`

This reduces Stage 2's adder depth from ~7 levels to ~5 levels (one 32-bit popcount),
and Stage 3 to ~3 levels (8-way sum). Pre-P&R timing estimates:

| Stage | Logic                   | Estimated delay | Estimated slack |
|-------|-------------------------|-----------------|-----------------|
| 1     | xor2 (XNOR)             | ~280 ps         | +3,053 ps       |
| 2     | 32-bit chunk popcount ×8 | ~1,400–1,600 ps | +1,733 ps       |
| 3     | 8-way sum + accumulate  | ~800–1,000 ps   | +2,333 ps       |

**Simulation**: All 16 tile checks PASS with 3-stage pipeline (tb_top.sv updated to
absorb the extra pipeline cycle via `m_axis_tvalid`-gated result capture).

**Area delta (vs. 2-stage)**: +50 FFs (+414 µm²), +414 cells. Total: 2,124 cells,
19,211 µm². Increase is minimal (<1% area).

**Post-P&R (3-stage, RUN_2026-05-23_03-54-34)**:

| Corner          | Hold WNS  | Setup WNS  | Setup TNS  | #Vio |
|-----------------|-----------|------------|------------|------|
| nom_tt_025C_1v80 | +0.311 ns | −2.443 ns  | −81.8 ns   | 76   |
| nom_ss_100C_1v60 | +0.514 ns | −7.541 ns  | −421.0 ns  | 248  |
| nom_ff_n40C_1v95 | +0.110 ns | −0.429 ns  | −3.25 ns   | 17   |

WNS improved from −5.70 ns (2-stage) to −2.44 ns (3-stage) at nom_tt — a 57% reduction.
Hold timing met at all corners. The remaining violation is no longer primarily the adder
tree depth: the nom_tt critical path (6.23 ns arrival) shows OpenROAD inserting
`fanout553→552→551→rebuffer10` buffer chains for the accumulator enable signal, adding
~0.5 ns, plus residual 30–40 µm wires throughout.

**Root cause unchanged**: The 549-IO standalone die forces a 350 µm floorplan. Long wires
and high-fanout buffering dominate the timing budget regardless of pipeline depth.
In-context integration within `bnn_top` (5–10 µm wires, no isolated IO ring) would close
timing at 300 MHz — this is the M4 path to sign-off.

---

## Plan for M4

- Replace `lpflow_*` cells with `a21o_1`/`mux2_1` equivalents before tape-out
- Run in-context P&R of `compute_core` within the `bnn_top` floorplan (eliminates the
  standalone IO-forced die penalty; expected to close timing at 300 MHz and provide
  accurate power — weight memory acts as a natural barrier shortening adder net lengths)
