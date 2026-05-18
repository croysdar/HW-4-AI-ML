# Project Scope Assessment

**Updated:** 2026-05-17 (post-CF07 synthesis)

## Current Scope

BNN hardware accelerator chiplet performing XNOR-popcount inference for Conv2–Conv4 layers of
a wildlife-camera CNN. Key design parameters: 256-bit 1-bit XNOR vectors, 300 MHz target,
Sky130A process, AXI4-Stream interface.

## Synthesis Result (CF07)

First synthesis run on `compute_core.sv` against Sky130 HD (tt_025C_1v80) revealed:

- **Total cell area:** 13,347.8 µm²  
- **Critical path:** 4057 ps (min-load) / ~5100 ps (with wire loads) for the 14-level
  XNOR → popcount → accumulate chain  
- **Timing status:** **VIOLATED** — 300 MHz target requires ≤3330 ps; flat combinational
  implementation misses by ~857 ps (min) to ~1800 ps (realistic)

## Scope Adjustment

The flat combinational XNOR-popcount architecture cannot meet 300 MHz in Sky130 HD without
pipelining. Scope is **adjusted** as follows:

1. **Retained:** 256-bit XNOR-popcount compute engine, 300 MHz target frequency, AXI4-Stream
   interface, Sky130A process, project focus on Conv2–Conv4 binary layers.

2. **Added:** One pipeline register stage between the XNOR output and the popcount reduction
   tree. This adds 1 cycle of latency per tile but keeps throughput at 1 tile/clock and brings
   the critical path within the 3.33 ns budget (~2.3 ns per half-path at min-load).

3. **Deferred to post-M3:** Full P&R (placement, routing, DRC/LVS) requires OpenROAD on a
   Linux host; M3 will deliver the RTL + synthesis report only, consistent with the milestone
   requirements.

## Rationale

The synthesis result is the first concrete evidence that pipelining is mandatory, not optional.
The decision is grounded in specific numbers: a 4057 ps path vs. 3330 ps budget, with the
dominant bottleneck (9 levels of xnor3_1 at ~326 ps each = 2934 ps) confirmed from Liberty
timing arcs. Inserting the pipeline register eliminates more than half of that bottleneck.
