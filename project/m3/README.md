# Milestone 3 — BNN Accelerator: Integration, Co-Simulation & Synthesis

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

## Tool Versions (reproducibility)

| Tool             | Version                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| Simulator        | Icarus Verilog 13.0 (stable, v13_0) — installed via Homebrew              |
| Waveform viewer  | GTKWave                                                                   |
| Synthesis + P&R  | OpenLane v2.3.10 (Docker: `ghcr.io/efabless/openlane2:2.3.10`)            |
| Image digest     | `sha256:37c3bd4ea0534a276cb2deb88d601044857bad2807b9bc5b36efe9d02c62624e` |
| Yosys (in image) | 0.46 (git sha1 e97731b9, compiled with pyosys)                            |
| PDK              | Sky130A HD (`sky130_fd_sc_hd__tt_025C_1v80`)                              |

No environment variables required. Docker daemon must be running.

---

## Deliverables Overview

### RTL (`rtl/`)

| File                                       | Description                                                                                                                                                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [rtl/compute_core.sv](rtl/compute_core.sv) | Pipelined BNN compute engine: XNOR + popcount accumulator with one pipeline register inserted between the XNOR stage and adder tree to meet 300 MHz timing                                                  |
| [rtl/top.sv](rtl/top.sv)                   | `bnn_top` integration module: AXI4-Stream slave (activation input), AXI4-Stream master (result output), 1,512×256-bit on-chip weight register file, runtime-configurable tile size via `cfg_beats_per_tile` |
| [rtl/interface.sv](rtl/interface.sv)       | AXI4-Stream interface definition (skid buffer helper) used by `bnn_top`                                                                                                                                     |

### Testbench (`tb/`)

| File                         | Description                                                                                                                                                                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [tb/tb_top.sv](tb/tb_top.sv) | End-to-end co-simulation testbench: drives AXI4-Stream activations through `bnn_top`, verifies dot-product results against a pure-SystemVerilog reference model (`sv_dot`), exercises conv2/conv3/conv4 tile sizes, AXI4-Stream backpressure, and weight reload |

### Simulation Outputs (`sim/`)

| File                                             | Description                                                                           |
| ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| [sim/cosim_run.log](sim/cosim_run.log)           | Icarus Verilog simulation log — shows all tile checks PASS against `sv_dot` reference |
| [sim/cosim_run.vcd](sim/cosim_run.vcd)           | VCD waveform dump from the full co-simulation run                                     |
| [sim/cosim_view.gtkw](sim/cosim_view.gtkw)       | GTKWave save file pre-zoomed to Phase 1, Tile 1 for quick waveform review             |
| [sim/cosim_waveform.png](sim/cosim_waveform.png) | Screenshot of GTKWave showing AXI4-Stream handshake and pipeline latency              |

### Synthesis (`synth/`)

| File                                                                             | Description                                                                                                                        |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [synth/config.json](synth/config.json)                                           | OpenLane 2 configuration: `compute_core` target, Sky130A HD, 300 MHz (3.33 ns period), 350×350 µm absolute die, 30% target density |
| [synth/synthesize_core.tcl](synth/synthesize_core.tcl)                           | Yosys TCL script for standalone `compute_core` synthesis with ABC `-D 3333` timing constraint (pre-P&R)                            |
| [synth/synthesize.tcl](synth/synthesize.tcl)                                     | Yosys TCL script for full `bnn_top` synthesis attempt (failed — see `synthesis_notes.md` scope decision)                           |
| [synth/compute_core_pipelined_netlist.v](synth/compute_core_pipelined_netlist.v) | Gate-level netlist output from standalone Yosys synthesis of pipelined `compute_core`                                              |
| [synth/timing_report.txt](synth/timing_report.txt)                               | Timing summary: pre-P&R Yosys/ABC estimate, plus post-routing OpenSTA WNS/TNS at all corners                                       |
| [synth/area_report.txt](synth/area_report.txt)                                   | Cell count and area: pre-P&R Yosys breakdown, plus post-P&R die/core/instance area                                                 |
| [synth/critical_path.md](synth/critical_path.md)                                 | Critical path analysis: M2 baseline, M3 pre-P&R two-stage breakdown, M3 post-P&R cell-remapped path                                |
| [synth/power_report.txt](synth/power_report.txt)                                 | OpenSTA post-routing power at nom_tt corner (293 mW total, with wire-load correction analysis)                                     |
| [synth/openlane_run.log](synth/openlane_run.log)                                 | OpenLane invocation history: Homebrew Yosys failure root cause + Docker resolution                                                 |
| [synth/compute_core_synth.log](synth/compute_core_synth.log)                     | Standalone Yosys synthesis log for `compute_core` (pre-P&R)                                                                        |
| [synth/yosys_synth_raw.log](synth/yosys_synth_raw.log)                           | Raw Yosys output from full `bnn_top` synthesis attempt (ABC hung at ~387K FFs)                                                     |
| [synth/yosys_synth_full.log](synth/yosys_synth_full.log)                         | Extended Yosys log including ABC timing output (pre-P&R)                                                                           |
| [synth/runs/RUN_2026-05-22_01-54-18/](synth/runs/)                               | OpenLane 2 P&R run that completed DRC/LVS clean + produced GDS at step 56-magic-streamout                                          |

### Top-Level (`m3/`)

| File                                                                                                 | Description                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [README.md](README.md)                                                                               | This file — M3 deliverables catalog, tool versions, key results, reproduction commands                                                                                                     |
| [synthesis_notes.md](synthesis_notes.md)                                                             | Synthesis narrative: scope decision, M2 timing violation root cause, M3 pipeline fix analysis, cell count impact, `lpflow_*` anomaly, OpenLane failure documentation, M4 plan (≥500 words) |

---

## Key Results

### Timing

Two stages of analysis: standalone Yosys/ABC pre-P&R (Liberty min-load arcs only) and
full OpenLane post-routing STA with extracted parasitics. Numbers below are at nom_tt/25°C/1.8V.

The M3 RTL uses a **3-stage pipeline** (XNOR → chunk popcount → final sum + accumulate).
The 2-stage version was synthesized first; its post-P&R result is retained for comparison.

| Stage of analysis              | Critical path                                             | Delay           | Slack       | Status       |
| ------------------------------ | --------------------------------------------------------- | --------------- | ----------- | ------------ |
| M2 pre-P&R (flat)              | PI → 14-level adder → FF(accum_out)                       | 4,057 ps        | −827 ps     | VIOLATED     |
| M3 2-stage pre-P&R Stage 1     | PI → xor2 → FF(xnor_reg)                                 | ~280 ps         | +3,053 ps   | MET          |
| M3 2-stage pre-P&R Stage 2     | FF(xnor_reg) → 7-level adder → FF                        | ~2,100–2,300 ps | +1,033 ps   | MET          |
| M3 2-stage post-P&R            | FF(xnor_reg[0]) → 16-level adder → FF(accum_out)         | ~9.0 ns         | **−5.7 ns** | **VIOLATED** |
| M3 3-stage pre-P&R Stage 1     | PI → xor2 → FF(xnor_reg)                                 | ~280 ps         | +3,053 ps   | MET          |
| M3 3-stage pre-P&R Stage 2     | FF(xnor_reg) → 5-level chunk popcount → FF(chunk_sums_r) | ~1,400–1,600 ps | +1,733 ps   | MET          |
| M3 3-stage pre-P&R Stage 3     | FF(chunk_sums_r) → 3-level sum → FF(accum_out)           | ~800–1,000 ps   | +2,333 ps   | MET          |
| **M3 3-stage post-P&R**        | FF(chunk_sums_r) → fanout buffers → FF(accum_out)        | 6.23 ns         | **−2.44 ns**| **VIOLATED** |

The 3-stage post-P&R WNS improved significantly vs. 2-stage (−2.44 ns vs. −5.70 ns, a 57%
reduction). The remaining violation is dominated by high-fanout rebuffering chains inserted
by OpenROAD for the accumulator enable/clear signals (fanout553→552→551→rebuffer10 chains
adding ~0.5 ns), plus residual 30–40 µm wire delays on the 350×350 µm standalone die.
The standalone floorplan cannot close timing due to the 549 IO pin constraint forcing a
large die and long internal wires. In-context P&R within `bnn_top` would have 5–10 µm
wires and is expected to close. See synthesis_notes.md for full analysis.

### Area

| Stage                      | Cells  | FFs | Cell Area  | Die Area    |
| -------------------------- | ------ | --- | ---------- | ----------- |
| M2 (Yosys pre-P&R)         | 1,273  | 31  | 13,348 µm² | n/a         |
| M3 2-stage (Yosys pre-P&R) | 1,710  | 289 | 19,077 µm² | n/a         |
| M3 2-stage post-P&R        | 3,747¹ | 289 | 32,714 µm² | 122,500 µm² |
| M3 3-stage (Yosys pre-P&R) | 2,124  | 339 | 19,211 µm² | n/a         |

¹ Includes 5 antenna diodes, 1,537 tap cells, and post-resizer buffer/sizing additions.
Filler cells (8,193) excluded from "cells" but contribute to the 122,500 µm² die.

The +258 FF increase from M2 to M3 is accounted for exactly: 256 (`xnor_reg[255:0]`) + 1 (`s_valid_r`) + 1 (`accum_clear_r`).

### Power (Post-P&R, nom_tt/25°C/1.8V)

Full OpenLane 2 P&R completed (run `RUN_2026-05-22_01-54-18`). DRC and LVS clean.

| Group         | Total Power  |
| ------------- | ------------ |
| Sequential    | 4.47 mW      |
| Combinational | 283.9 mW     |
| Clock         | 5.04 mW      |
| **Total**     | **293.4 mW** |

Note: Combinational power is inflated by long wires in the 350×350 µm standalone die
required for 549 IO pins. In-context estimate (within `bnn_top`): ~15–20 mW.
See [synth/power_report.txt](synth/power_report.txt) for full analysis.

### Simulation

All tile checks PASS across five test phases:

- Phase 1: conv4 tile size (5 beats, 3 tiles)
- Phase 2: conv2 tile size (2 beats, 4 tiles)
- Phase 3: conv3 tile size (3 beats, 4 tiles)
- Phase 4: conv4 with AXI4-Stream backpressure (3 tiles)
- Phase 5: weight reload to all-ones, expect dot = 1,280 (verified)

---

## How to Reproduce

### Simulation

```bash
cd project/m3
iverilog -g2012 -o sim/tb_top tb/tb_top.sv rtl/compute_core.sv rtl/top.sv rtl/interface.sv
vvp sim/tb_top | tee sim/cosim_run.log
```

### Synthesis (standalone Yosys)

```bash
cd project/m3/synth
yosys synthesize_core.tcl
```

### OpenLane 2 (Docker — required)

```bash
docker pull ghcr.io/efabless/openlane2:2.3.10
cd project/m3/synth
docker run --rm -v $(pwd)/..:/work -w /work/synth \
  ghcr.io/efabless/openlane2:2.3.10 python3 -m openlane config.json
```

Note: Homebrew Yosys (0.64/0.65) lacks pyosys support required by OpenLane 2.3.10.
Docker image contains Yosys 0.46 with pyosys — use this flow.
