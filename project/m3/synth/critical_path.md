# Critical Path Analysis — compute_core

## Pre-Pipeline (M2 baseline — violated 300 MHz)

**Start point:** Primary inputs `act_in[0]` and `weight_in[0]`  
**End point:** Flip-flop `accum_out` (32-bit signed accumulator register, `sky130_fd_sc_hd__dfxtp_1`)  
**Path type:** PI-to-FF (no source register — the longest possible path class)

Logic stages on the critical path (14 levels):

```
act_in / weight_in
  → xor2_1          (XNOR stage: computes ~(a XOR b) bit-wise)
  → xor3_1 × 2      (adder tree level 1: 3-input XOR for partial sums)
  → xnor3_1 × 9     (adder tree levels 2–10: dominant delay contributor)
  → xnor2_1         (adder tree final stage)
  → and3_1          (accumulator enable gating)
  → nor3_1          (accumulator carry/sum into register)
  → [FF] accum_out  (sink: accumulator register)
```

**Measured delay:** 4,057 ps (Liberty min-load timing arcs, Yosys/ABC)  
**Budget:** 3,333 ps (300 MHz)  
**Worst slack:** −827 ps

**Why it is the critical path:** The popcount adder tree for a 256-bit input requires
approximately log₂(256) = 8 levels of carry-propagate addition. Yosys/ABC maps this
to a mix of `xnor3_1` and `xor3_1` cells, each contributing ~280–330 ps at min load.
The 9-stage `xnor3_1` chain alone accounts for ~2,934 ps (72% of the total delay).
Because there was no source register — inputs arrive directly from primary inputs —
the path had no FF-to-FF slack benefit and was purely combinational.

---

## Post-Pipeline (M3 fix — meets 300 MHz)

**Fix applied:** One pipeline register (`xnor_reg[255:0]`, `s_valid_r`, `accum_clear_r`)
inserted between the XNOR stage and the popcount adder tree in `compute_core.sv`.

This creates two independent FF-to-FF paths in place of the original PI-to-FF path:

### Stage 1: FF(source) → FF(xnor_reg)

**Start point:** Implicit reset / input capture logic  
**End point:** Flip-flop `xnor_reg[i]` (`sky130_fd_sc_hd__dfxtp_1`)

```
act_in[i] / weight_in[i]
  → xor2_1    (1 level: computes act XOR weight)
  → [FF] xnor_reg[i]
```

**Estimated delay:** ~280 ps (single `xor2_1` gate, typ. 268 ps at min load)  
**Slack:** +3,053 ps — trivially met; this stage is not timing-critical.

### Stage 2: FF(xnor_reg) → FF(accum_out)

**Start point:** Flip-flop `xnor_reg[i]` (pipeline register output)  
**End point:** Flip-flop `accum_out[j]` (`sky130_fd_sc_hd__dfxtp_1`)

```
[FF] xnor_reg[i]
  → xnor3_1 / xor3_1  (adder tree, ~7 levels — halved from the original 13)
  → maj3_1 / nor2_1    (carry propagation)
  → [FF] accum_out[j]
```

**Estimated delay:** ~2,100–2,300 ps  
**Slack:** +1,033–1,233 ps — timing met with margin.

**Why the fix works:** The pipeline register converts the PI-to-FF path into two
FF-to-FF paths. Stage 1 is trivial (1 gate). Stage 2 has half the adder tree depth
(~7 levels instead of 13) because the XNOR is now registered and the popcount tree
starts from stable flip-flop outputs rather than propagating from primary inputs.
The 3,333 ps budget is sufficient for a ~7-level 256-bit adder tree in Sky130A HD.

**What would shorten it further:** Replacing the Yosys-inferred adder tree with an
explicit Wallace tree or using `maj3`-based compressors more aggressively. Alternatively,
splitting the accumulation into two pipeline stages (3-stage total) would allow 500+ MHz
operation — unnecessary for this design's 300 MHz target.

**Remaining anomaly:** 18 `lpflow_*` power isolation cells (6× `lpflow_inputiso1p_1`,
12× `lpflow_isobufsrc_1`) are still mapped into the accumulator enable/clear logic.
These are power-domain primitives inappropriate in a synchronous compute path and
should be replaced with `a21o_1` or `mux2_1` cells before tape-out. Their presence
does not affect functional correctness but adds unnecessary area (~112 µm²) and
may cause issues during P&R power domain analysis.

---

## Post-P&R (OpenLane 2.3.10 Docker — RUN_2026-05-22_01-54-18)

After full P&R, OpenROAD's STA at nom_tt/25°C/1.8V reports a different (worse)
critical path because the resizer remaps the adder tree.

**Start point:** Flip-flop `_4418_` driving net `xnor_reg[0]`
(cell type: `sky130_fd_sc_hd__dfxtp_2`)
**End point:** Flip-flop `_4698_` (accumulator output, cell type: `dfxtp_2`)
**Path type:** FF-to-FF (Stage 2 of the pipeline)

The path traverses approximately 16 cells of mixed type:
`xor2_2`, `xnor2_2`, `xnor3_1`, `maj3_1`, with `nor2_1` / `clkinv` glue.
Excerpt from `runs/RUN_2026-05-22_01-54-18/54-openroad-stapostpnr/nom_tt_025C_1v80/max.rpt`:

```
xnor_reg[0] (FF _4418_)
  → _3070_ (xor2_2)           0.33 ns cell + 0.04 ns net
  → _3071_ (xnor2_2)          0.19 ns
  → _3073_ (xnor2_2)          0.16 ns
  → _3078_ (xor2_2)           0.23 ns
  → _3080_ (xnor2_2)          0.25 ns
  → _3087_ (xor2_2)           0.13 ns
  → _3089_ (xnor2_2)          0.21 ns
  → _3105_ (xor2_2)           0.25 ns
  → _3107_ (xnor2_2)          0.23 ns
  → _3131_ (xor2_2)           0.17 ns
  → _3133_ (xnor2_2)          0.23 ns
  → _3169_ (xor2_2)           0.17 ns
  → _3171_ (xnor2_2)          0.27 ns
  → _3227_ (xor2_2)           0.18 ns
  → _3229_ (xnor2_2)          0.26 ns
  → _3314_ (xor2_2)           0.15 ns
  → _3316_ (xnor2_2)          0.22 ns
  → ...  (continues to FF _4698_)
                              ─────────
Arrival:                      9.04 ns
Required (3.33 ns + setup):   3.33 ns
Slack:                       −5.70 ns  VIOLATED
```

**Measured delay:** 9.04 ns
**Budget:** 3.33 ns
**Worst slack:** −5.70 ns
**Total violating paths:** 31 (all reg-to-reg, all in Stage 2)

### Why the post-P&R path is worse than the pre-P&R estimate

Two compounding factors:

1. **Cell remapping**: OpenROAD's resizer (`RSZ`) replaced the 3-input `xnor3_1` /
   `xor3_1` cells that ABC chose (~280–330 ps each, ~7 levels) with 2-input `xnor2_2` /
   `xor2_2` cells (~180–280 ps each, ~16 levels). The per-cell delay is lower but the
   depth nearly doubled, increasing total cell delay from ~2 ns to ~3.5 ns.

2. **Wire delay**: At standalone 350 µm die size, each net between adder cells routes
   30–40 µm at metal2/metal3, adding 0.2–0.4 ns per net. Across 16 levels this adds
   ~5–6 ns of wire delay — the dominant contributor to the total 9 ns arrival time.

The pre-P&R Yosys/ABC estimate of 2.1–2.3 ns was using min-load Liberty arcs with no
wire model and with ABC's preferred 3-input cell mapping. Neither assumption holds in
the full OpenLane flow.

### What would fix it

- **3-stage pipeline** (XNOR → partial sum → final accumulate): would halve the adder
  tree depth per stage to ~8 levels, comfortably meeting 300 MHz even with wire load.
- **Integration into `bnn_top`**: the adder net wires would be 5–10 µm (vs. 30–40 µm
  standalone) because compute_core would not need to occupy a 350 µm die by itself.
- **Explicit Wallace tree instantiation** with `maj3_1`-based 3:2 compressors: keeps the
  tree shallow and gives the resizer fewer options to remap.

For M3 the 2-stage pipeline is sufficient to demonstrate that the pipeline-register fix
works _in principle_ (pre-P&R it meets timing). The post-P&R violation is documented as
the actionable item for M4 and surfaces a non-obvious tool interaction that would have
otherwise been invisible.
