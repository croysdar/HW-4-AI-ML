# Option 5 Experiment — SRAM Macros with DRT_MIN_LAYER

**Status: pending. Run after the main WEIGHT_DEPTH=640 register-file synthesis completes.**

This experiment tries the OpenLane-equivalent of Innovus's
`setNanoRouteMode -routeBottomRoutingLayer met3` trick: force the router to use
met3+ only, so the SRAM macros' full-body met1+met2 obstructions stop blocking
signal routing.

## Hypothesis

The 5 prior failed runs (documented in `../README.md` and `../m3/synthesis_notes.md`)
all reported `GRT-0118: Routing congestion too high`. Investigation showed the
root cause was the SRAM macros' met1+met2 obstruction layers spanning the entire
macro body — TritonRoute had nowhere to put signals across the macro array.

OpenLane 2.3.10 exposes:
- `DRT_MIN_LAYER` — detailed routing minimum layer (analogous to Innovus's
  `routeBottomRoutingLayer`)
- `GRT_LAYER_ADJUSTMENTS` — per-layer global routing capacity reductions
- `RT_CLOCK_MIN_LAYER` — clock routing minimum layer

If we set:
- `DRT_MIN_LAYER = met3`
- `GRT_LAYER_ADJUSTMENTS = met1,1.0,met2,1.0` (100% blocked)
- `RT_CLOCK_MIN_LAYER = met3`

then signal nets stay above the macros' obstruction layers, and the router should
succeed even with macros present.

## What's in this directory

| File                    | Purpose                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `top_sram.sv`           | SRAM-backed `bnn_top` (8 banks × 512 entries × 32 bits)       |
| `config.json`           | OpenLane config with DRT_MIN_LAYER + manual macro placement   |
| `macro_placement.cfg`   | 4×2 grid placement in lower-left quadrant of 3600×3600 die    |
| `README.md`             | This file                                                     |

## Differences from register-file design

1. **Weight memory**: 8 SRAM macros instead of 163,840 flip-flops.
2. **WEIGHT_DEPTH**: 512 (one row of 8 macros) instead of 640. To match the
   register-file design's capacity we'd need a 2nd row of macros (16 total) —
   that's a follow-up if this minimal version succeeds.
3. **w_ptr→weight_word path**: now goes through SRAM with 1-cycle read latency,
   which naturally aligns with the +1 cycle through `axis_interface` skid buffer.
4. **Die size**: 3600 × 3600 µm (12.96 mm²). The SRAM array takes ~2.5 mm²
   (~19% of die area) leaving ~10 mm² for compute logic — plenty.
5. **Compute logic and AXI interfaces**: identical to the register-file design.

## Address space

```
w_addr / w_ptr ∈ [0, 511]   (9 bits)

Layer mapping (host responsibility):
  conv2 (128 words)         → addresses 0..127
  conv3 (384 words)         → addresses 128..511    (full row)
  conv4 cannot fit in 512   → would require 2nd row (16 macros, deferred)
```

For a tape-out demo, this is enough to prove the SRAM-routing approach works.
Scaling to 16 macros (1024 words) is a straightforward duplication of the
config — same flow, same techniques.

## How to launch

After the register-file WEIGHT_DEPTH=640 run completes (and its outputs are saved):

```bash
cd /Users/rebeccagilbert-croysdale/HW-4-AI-ML/project/m4/sram_experiment
docker run --rm -v $(pwd):/work -v $HOME/.volare:/root/.volare \
  -w /work ghcr.io/efabless/openlane2:2.3.10 \
  python3 -m openlane config.json > live_progress.log 2>&1 &
tail -f live_progress.log
```

Note the `-v $HOME/.volare:/root/.volare` mount — required so the SRAM PDK files
inside the container resolve to the host's volare cache.

## Expected outcomes

**Success** (best case): Global routing completes without GRT-0118.
- Detailed routing should follow (may need 1-2 iterations for antenna fixes).
- DRC/LVS may report issues with macro pin connections — fixable via `MACRO_PLACEMENT_CFG`
  tweaks or by adding `EXTRA_SDC_FILE` for macro-internal timing exceptions.

**Partial success**: Global routing succeeds but detailed routing fails on a
specific net.
- Usually fixable by enlarging `GRT_MACRO_EXTENSION` (currently 4).
- Or by adjusting macro spacing in `macro_placement.cfg`.

**Failure**: GRT-0118 again, despite the layer restrictions.
- Means TritonRoute's `DRT_MIN_LAYER` isn't enough to overcome the obstructions.
- Next steps: try `sky130_sram_1kbyte_1rw1r_32x256_8` (smaller footprint, smaller
  obstruction zones), or migrate to Innovus.

## Why this is the right next experiment

- **Low effort**: ~3 hours to set up, no RTL refactoring beyond what's done.
- **High information value**: tells us whether OpenLane is salvageable for SRAM
  macro flows, or whether Innovus is the only path. Either answer is useful.
- **Doesn't disturb the working register-file design**: this experiment lives in
  its own directory, with its own `top_sram.sv` and `config.json`. The main M4
  artifacts at `../rtl/`, `../synth/`, `../sim/` are unchanged.
