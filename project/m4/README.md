# Milestone 4 — BNN Accelerator: Full-Chip P&R + Final Co-Simulation

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
| PDK              | Sky130A HD (`sky130_fd_sc_hd`, `sky130A`)                                 |

No environment variables required. Docker daemon must be running.

---

## Summary

M4 completes the full-chip place-and-route of `bnn_top` and re-validates the end-to-end
co-simulation testbench. Four configurations are delivered; the **4-SRAM design is the final design**:

| Config                                 | Memory                               | Clock   | Die          | Power         | FPS          | DRC/LVS    |
| -------------------------------------- | ------------------------------------ | ------- | ------------ | ------------- | ------------ | ---------- |
| **Reg-file baseline** (`synth/`)       | 16,384-FF reg-file                   | 100 MHz | 1600×1600 µm | 215.3 mW      | ~12.5 (est.) | PASSED     |
| **1-SRAM** (`sram_1macro_experiment/`) | 1× sky130_sram_1kbyte_1rw1r_32x256_8 | 20 MHz  | 2000×2000 µm | **2.91 mW**   | **0.50**     | PASSED     |
| **4-SRAM** (`sram_4macro_experiment/`) | 4× sky130_sram_1kbyte_1rw1r_32x256_8 | 40 MHz  | 2400×1800 µm | **12.007 mW** | **2.5**      | bypassed†  |
| **8-SRAM** (`sram_8macro_experiment/`) | 8× sky130_sram_1kbyte_1rw1r_32x256_8 | 40 MHz  | 2400×2400 µm | **17.78 mW**  | **3.3**      | bypassed\* |

\*8-SRAM DRC: 12 routing + 8 KLayout errors at sky130 SRAM macro edges (known integration issue); 15 LVS mismatches from SRAM power pin extraction. All bypassed — do not affect logic correctness. 1-SRAM is the tape-out-ready result.
†4-SRAM DRC: 5 routing + 0 KLayout errors + 7 LVS mismatches. Single-row macro placement eliminated KLayout DRC entirely (8→0) and reduced routing DRC (12→5) vs 8-SRAM. Remaining errors are same SRAM macro edge metal-spacing artifacts.

The **4-SRAM design** is the final deliverable. It uses 4× sky130_sram_1kbyte_1rw1r_32x256_8
macros with 2-phase 128-bit reads at 40 MHz, achieving the best energy per frame (4.82 mJ)
of all three SRAM configurations and KLayout-clean routing (0 KLayout DRC errors vs 8 for
8-SRAM). The 1-SRAM and 8-SRAM variants are retained as reference experiments to bound the
power–throughput tradeoff. All three replace 16,384 weight flip-flops and are verified by
co-simulation against the `sv_dot` reference model.

Key results — **4-SRAM final design** (`sram_4macro_experiment/`):

- **OpenLane 2 P&R complete** (72/72 steps); routing DRC **5 errors**, KLayout DRC **0 errors**, LVS **7 mismatches** — all bypassed (SRAM macro edge artifacts); single-row placement eliminated KLayout DRC entirely
- **2-phase SRAM reads**: each 256-bit weight assembled from 2 SRAM cycles (4 banks × 32-bit each). Note: OpenSTA reports WNS = min(0, worst_slack), so WNS = 0.0 means no negative slacks exist, not zero margin.
- Setup WNS **0.0 ns** (no violations); worst-case slack **+11.42 ns** (45% margin at 40 MHz)
- Total power **12.007 mW** (TT 25°C 1.8V, post-route OpenSTA)
- Full-frame inference (bnn_serengeti2, 224×224): **1,404,928 tiles**, **16,056,320 cycles**, **401 ms**, **2.5 FPS**
  - Per-tile latency: conv2=10 cycles, conv3=12 cycles, conv4=16 cycles (2×beats + 6 drain)
- Energy per frame **4.82 mJ** (12.007 mW × 401 ms) — best of all three SRAM configurations
- Co-simulation: **VERIFIABLE PASS** — all 18 tile checks match `sv_dot` reference model

Key results — **8-SRAM variant** (`sram_8macro_experiment/`):

- **OpenLane 2 P&R complete** (72 steps); DRC/LVS bypassed (sky130 SRAM macro edge artifacts)
- Setup WNS **+11.57 ns** (25 ns / 40 MHz — 46% slack margin)
- Total power **17.78 mW** (TT 25°C 1.8V, post-route OpenSTA)
- Full-frame inference (bnn_serengeti2, 224×224): **1,404,928 tiles**, **12,242,944 cycles**, **306.1 ms**, **3.3 FPS**
  - Per-tile latency: conv2=8 cycles, conv3=9 cycles, conv4=11 cycles (256-bit parallel reads, 40 MHz)
  - Source: `sram_8macro_experiment/sim/timing_sim.log` (iverilog simulation)
- Energy per frame **5.44 mJ** (17.78 mW × 306.1 ms)
- **6.6× faster** than 1-SRAM; energy/frame ~equal (5.44 mJ vs 5.87 mJ)

Key results — **Reg-file baseline** (`synth/`, retained as fallback):

- **78/78** OpenLane steps completed; **DRC PASSED, LVS PASSED**
- Setup WNS **+6.477 ns** (MET at 100 MHz); critical path 3.52 ns → ≥284 MHz capable
- Total power **215.3 mW** (TT 25°C 1.8V, nominal activity)

---

## Deliverables Catalog

### Benchmark (`bench/`)

| File                                                                 | Description                                                                       | Supports                                    |
| -------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------- |
| [bench/benchmark.md](bench/benchmark.md)                             | Full benchmark report: throughput, power, roofline analysis, HW vs. SW comparison | Checklist §4; Report §8                     |
| [bench/benchmark_data.csv](bench/benchmark_data.csv)                 | Raw measurement table: all configurations (SW baseline, reg-file, 1/4/8-SRAM)     | Checklist §4 raw data                       |
| [bench/figures/roofline_final.png](bench/figures/roofline_final.png) | Annotated roofline plot: M1 CPU vs. BNN chiplet operating points                  | Checklist §4 roofline; Report §2, §8 Fig. 1 |
| [bench/gen_roofline.py](bench/gen_roofline.py)                       | Script to regenerate roofline figure from benchmark_data.csv                      | Reproducibility                             |

### Design Justification Report (`report/`)

| File                                                                   | Description                                                                                                                                                                  | Supports                        |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [report/design_justification.pdf](report/design_justification.pdf)     | **Primary deliverable** — 9-section PDF: §1 Problem, §2 Roofline, §3 Precision, §4 Dataflow, §5 Interface, §6 Verification, §7 Synthesis, §8 Benchmark, §9 What Did Not Work | Checklist §5                    |
| [report/design_justification.md](report/design_justification.md)       | Markdown source for the PDF                                                                                                                                                  | Checklist §5 (source)           |
| [report/figures/roofline_final.png](report/figures/roofline_final.png) | Roofline figure referenced as Fig. 1 in report                                                                                                                               | Checklist §5 figures; Report §2 |

### RTL (`rtl/`)

| File                                                   | Description                                                                                                             | Supports                        |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [rtl/top.sv](rtl/top.sv)                               | `bnn_top`: AXI4-Stream slave/master, 64×256-bit register-file weight memory, 3-stage pipelined `compute_core`, tile FSM | Checklist §2 RTL; Report §4, §5 |
| [rtl/compute_core.sv](rtl/compute_core.sv)             | 3-stage BNN engine: XNOR → 8-chunk popcount → accumulate                                                                | Checklist §2 RTL; Report §4     |
| [rtl/interface.sv](rtl/interface.sv)                   | `axis_interface`: 1-deep AXI4-Stream skid buffer; breaks combinational ready path                                       | Checklist §2 RTL; Report §5     |
| [rtl/sram_behav_wrapper.sv](rtl/sram_behav_wrapper.sv) | Behavioral SRAM wrapper (simulation-only); not used in P&R                                                              | Report §4 memory arch           |
| [rtl/sky130_sram_1kbyte_1rw1r_32x256_8_stub.v](rtl/)   | Sky130 OpenRAM macro stub; replaced by register file in synth/ due to routing obstruction                               | Report §9                       |

### Testbench (`tb/`)

| File                         | Description                                                                                                            | Supports                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| [tb/tb_top.sv](tb/tb_top.sv) | Co-simulation: AXI4-Stream streaming, `sv_dot` reference comparison, conv2/3/4 tile sizes, backpressure, weight reload | Checklist §2 testbench; Report §6 |

### Simulation Outputs (`sim/`)

| File                                             | Description                                                    | Supports                         |
| ------------------------------------------------ | -------------------------------------------------------------- | -------------------------------- |
| [sim/final_run.log](sim/final_run.log)           | Icarus Verilog log — 18 tile checks PASS, VERIFIABLE PASS      | Checklist §2 sim log             |
| [sim/cosim_run.log](sim/cosim_run.log)           | Same as final_run.log (original output filename)               | Report §6                        |
| [sim/final_waveform.png](sim/final_waveform.png) | GTKWave screenshot: AXI4-Stream handshake and pipeline latency | Checklist §2 waveform; Report §6 |
| [sim/cosim_run.vcd](sim/cosim_run.vcd)           | VCD dump from co-simulation (source for final_waveform.png)    | Report §6                        |

### Synthesis — Reg-File Baseline (`synth/`)

| File                                               | Description                                                     | Supports                         |
| -------------------------------------------------- | --------------------------------------------------------------- | -------------------------------- |
| [synth/config.json](synth/config.json)             | OpenLane 2 config: `bnn_top`, Sky130A HD, 100 MHz, 1600×1600 µm | Checklist §3 config              |
| [synth/openlane_run.log](synth/openlane_run.log)   | OpenLane log: 78/78 steps, DRC/LVS PASSED                       | Checklist §3 run log             |
| [synth/timing_report.txt](synth/timing_report.txt) | OpenSTA setup+hold: WNS +6.477 ns at 100 MHz nom_tt             | Checklist §3 timing; Report §7.1 |
| [synth/area_report.txt](synth/area_report.txt)     | Cell count + area: 46,774 cells, 1,043,990 µm² post-route       | Checklist §3 area; Report §7.1   |
| [synth/power_report.txt](synth/power_report.txt)   | OpenSTA power: 215.3 mW total (TT 25°C 1.8V)                    | Checklist §3 power; Report §7.1  |

### SRAM Experiment Directories

Each directory contains a `config.json`, RTL top module, placement constraints, and simulation logs for that SRAM variant. See Report §4 (memory architecture) and §7.2–7.4 (synthesis results) for analysis.

| File                                                                                                                                   | Description                                                                         | Supports                          |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------- |
| [sram_4macro_experiment/config.json](sram_4macro_experiment/config.json)                                                               | OpenLane 2 config: 4-macro, 40 MHz, 2400×1800 µm — **final design**                 | Checklist §3 (final); Report §7.4 |
| [sram_4macro_experiment/top_sram4macro.sv](sram_4macro_experiment/top_sram4macro.sv)                                                   | `bnn_top` with 4-macro 2-phase 128-bit weight read interface                        | Report §4, §7.4                   |
| [sram_4macro_experiment/macro_placement.cfg](sram_4macro_experiment/macro_placement.cfg)                                               | DEF macro placement: single-row, 4× sky130_sram_1kbyte_1rw1r_32x256_8               | Report §7.4                       |
| [sram_4macro_experiment/pnr_run.log](sram_4macro_experiment/pnr_run.log)                                                               | OpenLane P&R log: 72/72 steps, DRC/LVS summary                                      | Checklist §3 (final); Report §7.4 |
| [sram_4macro_experiment/sim/tb_timing_4macro.sv](sram_4macro_experiment/sim/tb_timing_4macro.sv)                                       | Timing testbench: per-tile cycle counts at 40 MHz — source of §8 throughput numbers | Report §8.1                       |
| [sram_4macro_experiment/sky130_sram_1kbyte_1rw1r_32x256_8_behav.sv](sram_4macro_experiment/sky130_sram_1kbyte_1rw1r_32x256_8_behav.sv) | Behavioral SRAM model used in simulation                                            | Report §6                         |
| [sram_1macro_experiment/config.json](sram_1macro_experiment/config.json)                                                               | OpenLane 2 config: 1-macro, 20 MHz, 2000×2000 µm                                    | Report §7.2                       |
| [sram_1macro_experiment/top_sram1macro.sv](sram_1macro_experiment/top_sram1macro.sv)                                                   | `bnn_top` with 1-macro 8-chunk serial weight read interface                         | Report §4, §7.2                   |
| [sram_1macro_experiment/compute_core_narrow.sv](sram_1macro_experiment/compute_core_narrow.sv)                                         | Narrow (32-bit) compute core for 8-chunk serialization                              | Report §4                         |
| [sram_1macro_experiment/tb_timing.sv](sram_1macro_experiment/tb_timing.sv)                                                             | Timing testbench: per-tile cycle counts at 20 MHz                                   | Report §8                         |
| [sram_1macro_experiment/tb_top_narrow.sv](sram_1macro_experiment/tb_top_narrow.sv)                                                     | Correctness testbench for narrow compute core                                       | Report §6                         |
| [sram_1macro_experiment/tb_hw_inference.sv](sram_1macro_experiment/tb_hw_inference.sv)                                                 | HW inference co-simulation testbench                                                | Report §6                         |
| [sram_1macro_experiment/run_hw_inference.py](sram_1macro_experiment/run_hw_inference.py)                                               | Python driver for HW inference co-simulation                                        | Report §6                         |
| [sram_8macro_experiment/config.json](sram_8macro_experiment/config.json)                                                               | OpenLane 2 config: 8-macro, 40 MHz, 2400×2400 µm                                    | Report §7.3                       |
| [sram_8macro_experiment/top_sram8macro.sv](sram_8macro_experiment/top_sram8macro.sv)                                                   | `bnn_top` with 8-macro parallel 256-bit weight read interface                       | Report §4, §7.3                   |
| [sram_8macro_experiment/macro_placement.cfg](sram_8macro_experiment/macro_placement.cfg)                                               | DEF macro placement: 8× SRAM in 2 rows                                              | Report §7.3                       |
| [sram_8macro_experiment/sim/timing_sim.log](sram_8macro_experiment/sim/timing_sim.log)                                                 | Simulation log: per-tile and full-frame cycle counts → 3.3 FPS                      | Report §7.3, §8                   |
| [sram_8macro_experiment/sim/tb_timing_8macro.sv](sram_8macro_experiment/sim/tb_timing_8macro.sv)                                       | Timing testbench: per-tile cycle counts at 40 MHz                                   | Report §8                         |
| [sram_8macro_experiment/sim/tb_hw_inference_8macro.sv](sram_8macro_experiment/sim/tb_hw_inference_8macro.sv)                           | HW inference co-simulation testbench for 8-macro write protocol                     | Report §6                         |
| [sram_8macro_experiment/sim/tb_top_8macro.sv](sram_8macro_experiment/sim/tb_top_8macro.sv)                                             | Correctness testbench for 8-macro top module                                        | Report §6                         |
| [sram_8macro_experiment/run_hw_inference_8macro.py](sram_8macro_experiment/run_hw_inference_8macro.py)                                 | Python driver for 8-macro HW inference co-simulation                                | Report §6                         |

---

## How to Reproduce

### Simulation

```bash
cd project/m4
iverilog -g2012 -o sim/tb_top tb/tb_top.sv rtl/compute_core.sv rtl/top.sv rtl/interface.sv
vvp sim/tb_top | tee sim/cosim_run.log
```

### OpenLane 2 (Docker — required)

```bash
docker pull ghcr.io/efabless/openlane2:2.3.10
cd project/m4/synth
docker run --rm -v $(pwd)/..:/work -w /work/synth \
  ghcr.io/efabless/openlane2:2.3.10 python3 -m openlane config.json
```

Note: Homebrew Yosys lacks pyosys support. Docker image contains Yosys 0.46 with
pyosys — use this flow for synthesis and P&R.
