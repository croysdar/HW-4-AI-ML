# Remaining Tasks Before M4

ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale

---

1. **Resolve M4 global routing congestion by reducing placement density from 50% to
   35% in `project/m4/synth/config.json` and re-running OpenLane 2.3.10 in Docker;
   the current run (`RUN_2026-05-23_14-39-00`) aborts at step 38 (`GRT-0118:
   Routing congestion too high`) because 24 SRAM macros consume ~83% of the 6400x2800 um
   die area at 50% density, leaving insufficient routing channels on metal2/metal3.**

2. **Replace the 18 `lpflow_*` power-isolation cells (`lpflow_inputiso1p_1` x6,
   `lpflow_isobufsrc_1` x12) in the `compute_core` accumulator enable/clear path by
   adding `"SYNTH_DONT_USE_CELL_LIST": ["sky130_fd_sc_hd__lpflow_*"]` to
   `project/m4/synth/config.json` before re-running synthesis; these cells are
   power-domain primitives that add ~112 um^2 of unnecessary area and may cause P&R
   power-domain analysis errors, but do not affect functional correctness.**

3. **Run a complete cocotb end-to-end simulation of one full conv3 layer (64->128,
   56x56 output) through `bnn_top` with the SRAM behavioral wrapper to measure actual
   cycle count, compute real attained GOPS, and convert the CF09 projected benchmark
   numbers in `codefest/cf09/benchmarks/benchmark_results.md` to measured values;
   the current M3 testbench only exercises individual tiles (conv4/conv2/conv3 single
   tiles), not a full layer pass with weight loading from SRAM.**
