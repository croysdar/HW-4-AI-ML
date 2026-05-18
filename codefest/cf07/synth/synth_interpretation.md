# Synthesis Interpretation — compute_core

**Design:** BNN XNOR-Popcount Compute Core  
**Tool:** Yosys 0.64 + ABC · Sky130A HD · tt_025C_1v80  
**Clock target:** 3.33 ns (300 MHz — derived from bandwidth requirement: 8.0 GB/s ÷ 32 bytes/cycle = 250 MHz minimum; 300 MHz adds 20% margin to reach 9.6 GB/s rated; see `project/m1/interface_selection.md`)

## Clock Period and Worst-Case Slack

ABC was constrained with `-D 3333` ps. Post-synthesis path analysis using Liberty timing arcs
at minimum load yields a critical path of **4057 ps**, exceeding the 3330 ps budget by
**827 ps** (worst-case slack **−827 ps**; larger with realistic wire loads). Timing is
**violated**; the flat combinational implementation cannot meet 300 MHz.

## Critical Path

The path runs from primary inputs (`act_in[0]`, `weight_in[0]`) to the accumulator register
— a PI-to-FF path with no source register. The 14-cell chain (PI → FF):

```
act_in / weight_in → xor2_1 → xor3_1 ×2 → xnor3_1 ×9 → xnor2_1 → and3_1 → nor3_1
→ dfxtp_1  [SINK: accum_out accumulator register]
```

Dominant cell types: **xnor3_1** (9 stages, 2934 ps combined) and **xor3_1** (3 stages,
~1029 ps) — 97% of combinational delay.

## Total Cell Area and Top Contributors

**Total: 13,347.8 µm²** (1,273 cells; 620.6 µm² sequential, 4.65%).

| Cell | Count | Area (µm²) | Share |
|---|---|---|---|
| `xnor3_1` | 119 | 2,680 | 20.1 % |
| `xor2_1`  | 297 | 2,600 | 19.5 % |
| `xor3_1`  |  98 | 2,330 | 17.5 % |
| `maj3_1`  | 221 | 2,210 | 16.6 % |

These four types consume 73.7% of total area.

## Warnings and Anomalies

**Verilator TIMESCALEMOD:** Sky130 black-box stubs lack `timescale`; cosmetic, not a design
defect.

**Unexpected power cells:** 13 `lpflow_inputiso1p_1` + 17 `lpflow_isobufsrc_1` (187.7 µm²)
were mapped into the accumulator enable/clear logic — inappropriate in a synchronous compute
path and should be replaced before tape-out.
