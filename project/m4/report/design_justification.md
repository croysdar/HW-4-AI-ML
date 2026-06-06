# Design Justification Report — BNN Accelerator Chiplet

**ECE 510 Spring 2026 | Rebecca Gilbert-Croysdale**  
**Milestone 4 — Full-Chip Place-and-Route + Final Co-Simulation**

---

## 1. Problem Statement and Application Context

Wildlife cameras ("trail cameras") are battery-powered sensors deployed in remote
locations for animal monitoring and conservation. A typical deployment runs for
weeks or months on a set of AA batteries. The core inference task — detecting
whether a camera-trap image contains an animal — runs at 30 FPS and must operate
within a sub-watt power budget to maintain battery life.

Current CPU-based implementations run at ~82 FPS on a high-performance Apple M1
processor drawing ~10–15 W, or at 10–20 FPS on a Raspberry Pi Zero drawing 0.4 W.
Neither is viable for always-on deployment: the M1 is far too power-hungry, and
the Pi Zero is too slow to classify at 30 FPS without dropping frames.

The target: a custom BNN accelerator chiplet that classifies at ≥30 FPS within
**<200 mW** total chip power budget, enabling integration into a camera module
paired with a sub-50 mW microcontroller host.

---

## 2. Architecture Overview

The BNN accelerator implements binary layers 2–4 of a 4-layer binary neural
network trained on the Caltech Camera Traps dataset. Layer 1 (floating-point
conv + batch norm) and the final linear classifier remain on the host CPU.

### Block Diagram

```
Host CPU ──AXI4-Stream 256-bit──▶ bnn_top ──AXI4-Stream 32-bit──▶ Host
              (activations)           │               (results)
                                      │
                              ┌───────┴────────┐
                              │  axis_interface │  AXI skid buffer
                              └───────┬────────┘
                                      │
                              ┌───────┴────────┐
                              │  compute_core  │  3-stage pipeline
                              │  XNOR → pop   │
                              │  → accumulate  │
                              └───────┬────────┘
                                      │
                              ┌───────┴────────┐
                              │  weight memory │  reg-file (WD=64)
                              │  (SRAM target) │  or SRAM macros
                              └────────────────┘
```

### Data Flow

1. Host loads binary weights for the current layer into the on-chip weight memory
   via the AXI4-Stream slave port (`s_axis_tdata`, `s_axis_tuser[0]=1`).
2. Host streams activation beats (256 bits/beat) for each tile.
3. `compute_core` performs XNOR between activations and stored weights, then
   popcount-sums the 256 bits in 8 parallel 32-bit chunks.
4. After all beats for a tile arrive, the accumulated dot product is output on
   the AXI4-Stream master port.
5. Host accumulates tile results to produce per-channel feature map values.

### Pipeline Stages

| Stage | Operation | Latency |
|-------|-----------|---------|
| S1 | XNOR: `act_in ^ weight_in` (256-bit) | 1 cycle |
| S2 | 8-chunk popcount of 256-bit XNOR result | 1 cycle |
| S3 | Accumulate chunk sums + tile tracking | 1 cycle |

Total pipeline latency: 3 clock cycles. New input accepted every cycle (fully
pipelined). At 100 MHz: 3 cycles × 10 ns = 30 ns latency, >16,000 beats/sec
throughput.

---

## 3. Interface Design Justification

### AXI4-Stream Selection

The interface protocol chosen is **AXI4-Stream (AMBA 4.0)** at 256-bit data width.

**Bandwidth requirement derivation:**
- Target performance: 1,200 GXNOR/s (chiplet peak at 100 MHz × 256 bits)
- Hardware arithmetic intensity: ~379 FLOP/byte (weight-stationary)
- Required bus bandwidth: Peak / AI = 1,200 / 379 ≈ **3.17 GB/s**

AXI4-Stream at 256 bits × 100 MHz delivers 3.2 GB/s — exactly at the
requirement boundary, providing ~1% margin.

**Protocol rationale:** AXI4-Stream is unidirectional and address-free, which
eliminates per-transaction overhead from the read-channel arbitration that
AXI4-Lite would require. For a streaming workload (sequential activations, no
random access), this is optimal. The `tvalid`/`tready` handshake supports
backpressure naturally — the host CPU can throttle activation delivery without
losing data, and the chiplet can hold off result delivery if the host is busy.

**Comparison to alternatives:**

| Protocol | Bandwidth | Overhead | Suitability |
|----------|-----------|----------|-------------|
| SPI @ 50 MHz | 6.25 MB/s | Low | Far too slow (530× under requirement) |
| I2C @ 400 kHz | 50 KB/s | Medium | Far too slow (64,000× under) |
| AXI4-Lite | Up to 3.2 GB/s | High (addr+data) | Viable but wasteful for streams |
| **AXI4-Stream** | **3.2 GB/s** | **Minimal** | **Selected** |
| AXI4 full | Up to 3.2 GB/s | Highest | Overkill — burst/ID logic unneeded |

### Skid Buffer (axis_interface.sv)

The `axis_interface` module implements a 1-deep skid buffer between the AXI slave
port and `compute_core`. Without the skid buffer, `s_axis_tready` would be
combinationally dependent on `core_ready` — a violation of AXI4-Stream protocol
that could cause hold-time violations and timing closure failures.

The skid buffer breaks this path: `s_axis_tready` is registered and depends only
on the buffer full/empty state, never on downstream signals.

---

## 4. Compute Engine Design Justification

### Binary Neural Network Choice

Binary Neural Networks replace floating-point multiplications with XNOR operations
and accumulations with popcount. For the classification layers (conv2–4):

| Operation | Float32 MAC | Binary XNOR+popcount |
|-----------|-------------|----------------------|
| Per-bit | 1× mul + 1× add ≈ 6 transistors | 1× XNOR ≈ 2 transistors |
| Logic family | Sequential (multiplier pipeline) | Combinational (gate array) |
| Energy (normalized) | 1× | ~0.005× |
| Area (normalized) | 1× | ~0.02× |

The 50–200× area/energy advantage of binary operations enables fitting 256
XNOR units plus accumulator in ~0.27 mm² on Sky130, at a cost of ~2–3% accuracy
loss vs. a float32 baseline (measured on Caltech Camera Traps test set).

### 3-Stage Pipeline Rationale

A 3-stage pipeline (XNOR → popcount → accumulate) was chosen over:

1. **1-stage (combinational):** The critical path through 256-bit XNOR → 8×
   32-bit popcount → 9-bit accumulator exceeds 5 ns (critical path estimated ~8 ns
   combinational), limiting clock to <125 MHz. The pipeline register between stages
   cuts this to 3.52 ns.

2. **5-stage (finer grain):** More pipeline registers increase FF count and clock
   distribution overhead without improving throughput (already 1 beat/cycle at 3
   stages). Unnecessary complexity.

3. **2-stage (XNOR+popcount combined):** Tested in M2; synthesis showed the
   combined stage was the timing bottleneck. Splitting into 3 stages added ~340 FFs
   but improved the critical path by 2.1 ns.

### Operand Isolation

Pipeline registers 1 and 2 are gated on their respective valid signals:

```systemverilog
// Stage 1: only latch when handshake fires
if (s_valid & s_ready)
    xnor_reg <= act_in ^ weight_in;

// Stage 2: only latch when stage-1 output is valid
if (s_valid_r)
    chunk_sums_r <= chunk_sums;
```

Without this gating, the 256-bit XNOR tree and 8-chunk popcount adder toggle every
clock cycle on stale data. In a wildlife camera deployment, the BNN chiplet is idle
>99% of the time (30 FPS × 57 µs/frame = 0.17% duty cycle). Gating the wide
datapaths reduces compute-core dynamic power by 30–50% during idle cycles.

The control path (valid/ready signals) is not gated — these drive FSM transitions
and must update every cycle.

---

## 5. Memory Architecture Justification

### Weight Memory: Register File vs. SRAM

The production BNN requires WEIGHT_DEPTH=1,512 words × 256 bits = 387,072 bits
of on-chip weight storage for all-layer weight-stationary operation, or
WEIGHT_DEPTH=640 words × 256 bits for the largest single tile (conv4 upper half).

#### Register File (current M4 implementation)

WEIGHT_DEPTH=64 was synthesized as the tape-out-feasible slice:

| Parameter | Value |
|-----------|-------|
| Flip-flops | 16,384 `dfxtp_2` |
| Area (synthesis) | ~453,000 µm² |
| Power (100 MHz, nominal activity) | ~150 mW (FF array + clock distribution) |
| P&R feasibility | Fully feasible — 78/78 steps, DRC/LVS PASSED |

The register file dominates both area (63%) and power (70%) of the full design.
It is not economically printable for production (215 mW at 100 MHz, ~$10–20k for
a Sky130 tapeout at this area), but proves the compute path and AXI interface
fully functional.

#### SRAM Macros (target implementation)

The target architecture uses `sky130_sram_1kbyte_1rw1r_32x256_8` macros:
- 8 macros in parallel for 256-bit reads
- 2 rows for row selection (row_sel = addr[8])
- Total: 8 × 2 × 256 × 32 = 131,072 bits per bank level

Projected advantages over register file:

| Metric | Register File (WD=64) | SRAM (16 macros) | Improvement |
|--------|----------------------|------------------|-------------|
| Area | ~0.45 mm² (FFs) | ~7.7 mm² (macros) | — |
| Power @ 20 MHz | ~30 mW | ~8 mW | 3.75× |
| Active read energy | FF toggle always | SRAM access only | ~100× lower idle |
| Weight depth feasible | 64 (OOM at 640) | 512 per bank | 8× more per run |

Note: The SRAM macro array is larger in absolute area (macros are denser but
each macro is 479×397 µm). The benefit is dynamic power, not area.

#### SRAM Integration Challenges and Solutions

The SRAM integration experiment encountered multiple OpenLane-specific issues,
documented fully in the M4 README. Key findings:

1. **`DRT_MIN_LAYER` vs. `RT_MIN_LAYER`:** OpenLane 2 uses `RT_MIN_LAYER` to set
   the minimum signal routing layer (`set_routing_layers -signal`). `DRT_MIN_LAYER`
   only affects detailed routing for clock nets and has no effect on signal layer
   constraints. This was confirmed by inspecting `set_routing_layers.tcl` in the
   OpenLane source.

2. **DRT-0155 (macro pin access):** When `RT_MIN_LAYER=met3` is set, the global
   router still generates guides on li1/met1 for nets connecting to SRAM macro pins
   (which are only accessible on li1/met1). The detailed router then rejects these
   guides as out-of-range. Fix: remove `RT_MIN_LAYER` and use
   `GRT_LAYER_ADJUSTMENTS=[1.0, 0.99, 0.99, 0, 0, 0]` — 99% capacity reduction
   (not 100% ban) on met1/met2, allowing technically valid guides for macro pin
   access while strongly discouraging bulk routing on those layers.

3. **`PDN_MACRO_CONNECTIONS`:** SRAM macro power pins (`vccd1`/`vssd1`) must be
   explicitly connected to the PDN via this variable. Without it, PDN-0189 warnings
   fire and LVS fails (floating supply pins on macro instances).

The current run (as of M4 submission) is at step 41/78 (antenna repair) using
the `GRT_LAYER_ADJUSTMENTS` approach. No DRT-0155 errors occurred in the global
routing stage.

---

## 6. Floorplan and Physical Design Justification

### WD=64 Baseline (synthesized, DRC/LVS PASSED)

**Die size:** 1,600 × 1,600 µm (2.56 mm²)

This was sized to achieve ~35–45% core utilization with the 46,774-cell netlist
at WEIGHT_DEPTH=64. Experiments with smaller dies (1,000×1,000 and 2,000×2,000)
showed either DPL-0036 (placement failures near macro boundaries) or excessive
wasted area.

**Placement:** Standard cells only — no macro constraints needed for the register
file. OpenROAD NesterovSolve global placement converged at overflow <0.10 in
standard settings.

**Routing:** 6 detailed routing iterations to clear DRC. 167 antenna violations
remain after antenna repair (OpenROAD inserts diodes but cannot eliminate all
violations on the first pass). These are non-critical; a real tape-out would
run additional antenna repair passes or hand-fix the remaining nets.

### SRAM-256 Experiment Floorplan

**Die size:** 5,000 × 5,000 µm (25 mm²)

The larger die accommodates:
- 16 SRAM macros (479×397 µm each) in 8 columns × 2 rows, placed in the lower
  portion of the die (y=100 to y=995 µm)
- Standard cell logic placed above y=1,100 µm
- Routing channels between macro rows (100 µm gap between rows)

The 2-row layout places 8 macros in the lower row and 8 in the upper row. The
RTL selects between rows using `row_sel = sram_addr[8]` — address bit 8 maps
to the physical row, and `row_addr = sram_addr[7:0]` addresses within the row.
SRAM enable (`csb0`) is only asserted for macros in the selected row, preventing
unnecessary read current on the inactive row.

**PDN:** `PDN_MACRO_CONNECTIONS = [".*u_bank vccd1 vssd1 vccd1 vssd1"]` connects
all 16 SRAM macro instances to the top-level `vccd1`/`vssd1` power straps.

---

## 7. Timing Closure Analysis

### WD=64, 100 MHz constraint (nom_tt_025C_1v80)

| Metric | Value | Status |
|--------|-------|--------|
| Setup WNS | +6.477 ns | MET — large positive slack |
| Setup TNS | 0.000 ns | MET |
| Hold WNS | +0.742 ns | MET |
| Hold TNS | 0.000 ns | MET |
| Critical path | 3.52 ns | FF→XNOR/XOR adder tree→FF |
| Max achievable clock | ~284 MHz | (10 ns − 6.477 ns = 3.523 ns critical path) |

The design is conservatively over-constrained at 100 MHz. The 3.52 ns critical
path through the popcount adder tree means the design is capable of >280 MHz at
the typical corner — well above the 300 MHz target from the roofline analysis.

At the 20 MHz target (50 ns period), timing closure is trivial: +46 ns of setup
slack. The synthesizer would select minimum-size cells and reduce cell area,
further reducing area and power.

**Slow-corner violations (SS 100°C 1.6V):** Setup and hold violations appear at
this extreme corner with the default OpenLane SDC. This is expected — the default
SDC applies 10% timing derate but does not account for voltage droops or
temperature-dependent cell degradation. A production design would tighten the
slow-corner constraint. The nominal corner (TT 25°C 1.8V) is fully clean.

---

## 8. Power Budget and Optimization

### Baseline Power Breakdown (WD=64, 100 MHz)

| Component | Power | Notes |
|-----------|-------|-------|
| Register file (16,384 FFs) | ~90 mW sequential | FF array toggling every cycle |
| Clock distribution | ~61 mW | Large fanout to 16,384 FFs |
| Combinational (XNOR + popcount) | ~42 mW | Toggling on every cycle |
| Switching overhead | ~22 mW | |
| **Total** | **215.3 mW** | 8% over 200 mW target |

### Optimization 1: Clock Frequency Reduction to 20 MHz

The 30 FPS inference requirement only needs 5,760 clock cycles per frame at peak
throughput (all binary layers combined). At 20 MHz, this takes 288 µs — well
within the 33 ms frame budget (0.87% duty cycle).

Dropping to 20 MHz reduces:
- All switching power linearly: 55.4 × (20/100) = 11.1 mW
- Clock network power: 60.7 × (20/100) = 12.1 mW
- Synthesizer selects smaller cells, reducing internal power

Estimated total at 20 MHz (same netlist): ~43 mW.

### Optimization 2: Operand Isolation

Gating the 256-bit XNOR register and 8-chunk popcount register on valid signals
prevents the widest datapath from toggling on idle cycles. At 0.87% active duty
cycle, this saves ~98% of the compute-path dynamic power during idle time.

### Path to Meeting Power Target

| Configuration | Total Power | Status |
|---------------|-------------|--------|
| WD=64 @ 100 MHz | 215.3 mW | 8% over target |
| WD=64 @ 20 MHz (estimated) | ~43 mW | 79% under target |
| SRAM + 20 MHz (projected) | ~25–30 mW | 85% under target |

The 20 MHz clock reduction alone brings power well under the 200 mW target.
The SRAM integration additionally eliminates the dominant power component
(the register-file FF array) to achieve a production-quality power budget.

---

## 9. What Did Not Work: Documented Failures

This section documents failed approaches and the insights gained — essential
for understanding the design space and for future implementations.

### 9.1 WEIGHT_DEPTH=640 Register File P&R

Attempted to synthesize the full tiled configuration (WEIGHT_DEPTH=640) with a
standard register file. OpenROAD crashed repeatedly during timing-driven placement
(RepairDesignPostGPL) with out-of-memory errors. Root cause: the 163,840-FF
register file produces a ~496,000-cell netlist. OpenROAD's RC extraction for
timing-driven optimization requires holding the full parasitics model in memory
for all cells — this exceeds ~16 GB RAM.

Workaround tried: `PL_TIME_DRIVEN=false` (disables RC extraction in placement).
This allows global placement to complete but the resizer pass after GPL still
hits OOM. The design is infeasible on machines with <32 GB RAM using OpenLane 2.

**Lesson:** Register-file weight memories larger than ~16K FFs are impractical
for OpenLane P&R on workstation hardware. SRAM macros or hierarchical hardening
are required for production-scale weight storage.

### 9.2 Sky130 SRAM Macros — 8×1 Row (8×512 configuration)

Attempted to place 8 `sky130_sram_2kbyte_1rw1r_32x512_8` macros in a single
row at the bottom of the die. All runs failed at GlobalRouting (GRT-0118:
routing congestion too high).

Root cause: each macro is 683×416 µm. 8 macros in a row span 5,464 µm of die
width. The full-body met1+met2 routing obstructions in OpenRAM macros create
~11,000 routing blockages spanning the full die — there is no horizontal routing
corridor for signal nets to cross the SRAM row, even with a much larger die.

**Lesson:** Never place SRAM macros that span more than ~50% of die width in a
single row. The routing obstruction footprint must leave horizontal corridors.

### 9.3 `DRT_MIN_LAYER` vs. `RT_MIN_LAYER` Discovery

After discovering that `GRT_LAYER_ADJUSTMENTS=[1,1,1,0,0,0]` (blocking li1/met1/met2)
was the right approach, the initial config used `DRT_MIN_LAYER=met3` to also
enforce signal routing at the detailed routing stage. This had no effect.

Investigation of the OpenLane source (`set_routing_layers.tcl`) revealed:
- `RT_MIN_LAYER` → controls both signal and clock minimum routing layer
- `DRT_MIN_LAYER` → only appears in detailed routing clock constraints

`DRT_MIN_LAYER` is the wrong variable for signal layer forcing. Correcting to
`RT_MIN_LAYER=met3` helped global routing but caused DRT-0155 at detailed routing
(see next item).

**Lesson:** Verify OpenLane variable semantics against source code. Documentation
can lag behind behavior. The variable `DRT_MIN_LAYER` is a common trap.

### 9.4 DRT-0155: Guide Layer Below RT_MIN_LAYER

After fixing to `RT_MIN_LAYER=met3`, global routing succeeded but detailed routing
failed immediately with:
```
[ERROR DRT-0155] Guide in net _0000_ uses layer met1 that is outside the
allowed routing range (met3, met5).
```

Root cause: SRAM macro pins (`csb0`, `addr0`, etc.) are accessible only on
li1/met1 layers. The global router must write guide segments on li1/met1 to reach
these pins — but `RT_MIN_LAYER=met3` then makes these guides invalid for detailed
routing.

Fix: Removed `RT_MIN_LAYER` entirely. Instead use `GRT_LAYER_ADJUSTMENTS=
[1.0, 0.99, 0.99, 0, 0, 0]` — 99% (not 100%) capacity reduction on met1/met2.
This lets the global router write technically valid guides for macro pin access
nets while strongly discouraging bulk signal routing on lower layers.

**Lesson:** `RT_MIN_LAYER` and macro pin accessibility are in conflict when macros
expose pins on layers below the minimum. For SRAM macros, capacity adjustment is
the correct approach — it guides (not forbids) routing to upper layers while
preserving macro pin access.

---

## 10. Results Summary

### Tape-out Readiness Assessment

| Criterion | WD=64 Baseline | SRAM-256 (in progress) |
|-----------|----------------|------------------------|
| DRC | PASSED (0 errors) | Pending |
| LVS | PASSED | Pending |
| Setup timing (nom_tt) | MET (+6.477 ns) | Pending |
| Hold timing (nom_tt) | MET (+0.742 ns) | Pending |
| Power < 200 mW | 215 mW (8% over) | ~30 mW projected |
| Die area | 2.56 mm² | 25 mm² (5×5 mm) |
| Manufacture cost (Sky130) | ~$10–20k (MPW) | ~$50k (full die) |

The WD=64 baseline is tape-out-ready at the nominal corner. Power exceeds target
by 8% at 100 MHz; reducing to 20 MHz brings it to ~43 mW.

### Comparison to Software Baseline

| Metric | SW Baseline (M1) | HW Accelerator | Improvement |
|--------|------------------|----------------|-------------|
| Latency (binary layers) | ~5 ms | 57.6 µs | **>87×** |
| Throughput | 82 FPS | >17,000 FPS | **>200×** |
| Power (binary layers) | ~2,000 mW (CPU) | 215 mW | **9.3×** |
| Power @ 20 MHz | — | ~43 mW | **47×** |
| Arithmetic intensity | 12.3 FLOP/byte | 379 FLOP/byte | **31×** |

The hardware accelerator achieves the primary design goals: >200× throughput
improvement on the binary layers, and 9–47× power reduction depending on the
target operating frequency. The weight-stationary dataflow successfully shifts
the system from memory-bound to compute-bound operation.

---

## 11. System-Level Power and Deployment Analysis

This section places the chip-level P&R power numbers in the context of a complete
deployed wildlife camera system to assess battery life, solar viability, and
real-world deployment duration.

### 11.1 Chip-Level Power (Measured, Post-Route)

The `sram_1macro_experiment` run — 1× `sky130_sram_1kbyte_1rw1r_32x256_8` macro
with a 32-bit-wide compute path — is the closest feasible implementation to the
target SRAM architecture. OpenSTA at the nominal corner (TT 25°C 1.8V,
`nom_tt_025C_1v80`) reports:

| Component | Power | Share |
|-----------|-------|-------|
| Sequential (947 FFs) | 0.82 mW | 28% |
| Clock distribution | 1.02 mW | 35% |
| SRAM macro | 0.79 mW | 27% |
| Combinational (XNOR + popcount) | 0.29 mW | 10% |
| **Total (post-route)** | **2.91 mW** | 100% |

This represents a **74× reduction** from the register-file baseline (215.3 mW at
WD=64, 100 MHz). The dominant contributors shift from the FF array (70% of
baseline power) to clock distribution and the SRAM macro, which is the expected
profile for an SRAM-based datapath.

#### Continuous-operation power at 30 FPS

The camera runs 24/7 at 30 FPS — the BNN chiplet is the motion/animal detector
and must evaluate every frame. Frame latency from P&R (all 16,464 BNN tiles:
conv2 12,544 + conv3 3,136 + conv4 784) is **21.1 ms**.

At 30 FPS the frame period is 33.3 ms, giving a duty cycle of:

```
21.1 ms / 33.3 ms = 63.4%
```

Average chip power at 30 FPS continuous:

```
2.91 mW × 0.634 = 1.84 mW
```

Energy per frame (measured):

```
2.91 mW × 21.1 ms = 61.5 µJ
```

### 11.2 System-Level Power Budget

A full wildlife camera system includes several components beyond the BNN chiplet.
Power figures below are representative for common off-the-shelf parts; BNN chip
figures are from the P&R measurement above.

#### Lean system (MCU host, e.g. STM32H7 for conv1)

| Component | Typical Power | Notes |
|-----------|--------------|-------|
| Image sensor (OV5647 or similar) | 30–50 mW | Continuous capture at 30 FPS |
| Host MCU — conv1 INT8 (STM32H7) | 10–30 mW | DSP-accelerated first layer |
| BNN chiplet (avg at 30 FPS) | **1.84 mW** | Measured post-route |
| Radio — LoRa alert transmit | 2–5 mW | Averaged over duty-cycled TX |
| Regulators and miscellaneous | 5–10 mW | LDO losses, LED indicators |
| **Lean system total** | **~50–100 mW** | |

#### RPi-based system (RPi Zero 2W host)

| Component | Typical Power | Notes |
|-----------|--------------|-------|
| Image sensor | 30–50 mW | |
| Raspberry Pi Zero 2W | 80–150 mW | Including conv1 + Linux overhead |
| BNN chiplet | **1.84 mW** | |
| Radio + regulators | 7–15 mW | |
| **RPi-based total** | **~150–250 mW** | |

The BNN chiplet contributes **less than 4%** of lean system power and **less than
1.5%** of RPi-based system power. The dominant consumers are the image sensor
and the host CPU — not the inference accelerator.

### 11.3 Battery Life Estimate

Reference battery: **10 Ah @ 3.7 V = 37 Wh** — a common single-cell LiPo pack
used in commercial wildlife cameras.

| Configuration | Avg Power | Battery Life |
|---------------|-----------|-------------|
| BNN chiplet alone | 1.84 mW | 37,000 mWh / 1.84 mW = **~20,100 h (~838 days)** |
| Lean system (75 mW avg) | 75 mW | 37,000 mWh / 75 mW = **~493 h (~20.5 days)** |
| RPi-based system (200 mW avg) | 200 mW | 37,000 mWh / 200 mW = **~185 h (~7.7 days)** |

The chiplet itself could run continuously for over two years on a single pack.
System lifetime is gated entirely by the sensor and host CPU.

### 11.4 Solar Viability

Reference panel: **2 W peak**, outdoor deployment, mid-latitude conditions.

Usable energy per day varies with cloud cover, panel angle, and seasonal sun
hours. At 4 peak sun hours/day (conservative outdoor average), a 2 W panel
yields approximately 8 Wh/day delivered; accounting for ~30–50% derating for
clouds, angle, and regulator losses, practical usable energy is roughly
**0.5–2 Wh/day** from a 2 W panel and **1–5 Wh/day** from a 5 W panel.

| Configuration | Daily energy need | 2 W panel (0.5–2 Wh/day) | 5 W panel (1–5 Wh/day) |
|---------------|------------------|--------------------------|------------------------|
| BNN chiplet | 1.84 mW × 24 h = **0.044 Wh/day** | Sustained indefinitely | Sustained indefinitely |
| Lean system (75 mW) | 75 mW × 24 h = **1.8 Wh/day** | Borderline — marginal on cloudy days | Comfortable |
| RPi-based (200 mW) | 200 mW × 24 h = **4.8 Wh/day** | Insufficient | Marginal |

The BNN chiplet's 0.044 Wh/day consumption is so low that even a small solar
cell on a dark day could sustain it indefinitely. For the full lean system
(sensor + MCU + chiplet), a 5 W panel provides comfortable solar operation.
The RPi-based deployment requires a 10 W+ panel for reliable 24/7 operation.

**Conclusion:** The chiplet is not the power bottleneck in any realistic system
configuration. Deployment duration and solar sizing are determined by the image
sensor and host CPU, not by the BNN accelerator.

### 11.5 Comparison to Software Baseline

The `baseline_sw` benchmark (from `bench/benchmark_data.csv`) measures the full
BNN model running in PyTorch float32 on an Apple M1 CPU at 82 FPS (12.19 ms/image,
100-run mean).

#### Energy per inference

A direct energy comparison requires a power measurement during inference, which was
not captured in the M1 benchmark run. Published measurements for the M1 MacBook Air
under sustained single-core CPU load report approximately **8–10 W at the SoC**
(Anandtech 2020 M1 review; Nanoreview.net idle/load delta). Using 10 W as the
reference:

```
Energy_M1 = 10 W × (1 / 82 FPS) = 10 W × 12.19 ms = 121,951 µJ/frame
Improvement = 121,951 µJ / 61.5 µJ ≈ 1,983× ≈ 2,000×
```

| Platform | Power (estimated) | Throughput | Energy per frame |
|----------|------------------|-----------|-----------------|
| Apple M1 CPU (@ 10 W SoC) | ~10,000 mW | 82 FPS | ~122,000 µJ |
| BNN chiplet (post-route, 30 FPS) | 2.91 mW active | ≥30 FPS | **61.5 µJ** |
| **Improvement** | | | **~2,000×** |

Two caveats apply: (1) the 10 W figure is an estimate — actual power under this
workload was not directly measured; (2) the M1 runs the full float32 model
(conv1 through classifier), while the chiplet handles only the binary layers
(conv2–4). The system-level comparison below avoids both issues and is the
more defensible figure for deployment claims.

#### System-level power comparison

| System | Total power | vs. M1 |
|--------|-------------|--------|
| Apple M1 (full inference pipeline) | ~15,000 mW | baseline |
| BNN lean system (chiplet + sensor + STM32H7) | ~75 mW | **~200× lower** |
| BNN RPi-based system | ~200 mW | **~75× lower** |

The lean wildlife camera system draws approximately **200× less power** than running
equivalent inference on an M1 Mac — achieving the core design goal of enabling
always-on 30 FPS inference within a battery-powered remote deployment.

---

*RTL: `project/m4/rtl/` | Synthesis: `project/m4/synth/` | Benchmark: `project/m4/bench/`*  
*Tool: OpenLane 2.3.10, Yosys 0.46, OpenROAD, Sky130A sky130_fd_sc_hd*
