# CMAN — Sneak Paths in a 2×2 Resistive Crossbar

Cell resistances: R[0][0] = 1 kΩ, R[0][1] = 2 kΩ, R[1][0] = 2 kΩ, R[1][1] = 1 kΩ.

## (a) Ideal Read — I_col0

Conditions: V_row0 = 1 V, V_row1 = 0 V, V_col0 = 0 V, V_col1 = 0 V.

Only R[0][0] sees a voltage drop:

- I_col0 = 1 V / 1 kΩ = **1 mA**

## (b) Floating Node Voltages

Conditions: V_row0 = 1 V, V_col0 = 0 V; row 1 and col 1 floating.

The sneak path is a series chain row 0 → R[0][1] → col 1 → R[1][1] → row 1 → R[1][0] → col 0, with equivalent resistance 2 kΩ + 1 kΩ + 2 kΩ = 5 kΩ. The full 1 V drives this chain:

- I_sneak = 1 V / 5 kΩ = **0.2 mA**

Tracing voltage drops along the path:

- R[0][1] (2 kΩ): 0.2 mA × 2 kΩ = 0.4 V drop → **V_col1 = 1.0 − 0.4 = 0.6 V**
- R[1][1] (1 kΩ): 0.2 mA × 1 kΩ = 0.2 V drop → **V_row1 = 0.6 − 0.2 = 0.4 V**
- R[1][0] (2 kΩ): 0.2 mA × 2 kΩ = 0.4 V drop → 0.4 − 0.4 = 0 V

## (c) Actual I_col0 with Sneak Path Itemized

- Direct path (row 0 → R[0][0] → col 0): 1 V / 1 kΩ = **1.0 mA**
- Sneak path (row 0 → R[0][1] → col 1 → R[1][1] → row 1 → R[1][0] → col 0): 0.4 V / 2 kΩ = **+0.2 mA**
- **Total I_col0 = 1.2 mA**

## (d) How Sneak Paths Corrupt MVM Results

When unselected rows and columns are left floating, sneak paths form through neighboring cells and adds more current into the sense column that has nothing to do with the intended weights. This means the sensed current (1.2 mA) is higher than the true weighted sum (1.0 mA), producing an incorrect MVM result. In larger arrays this problem compounds because every undriven node adds more sneak paths, making the error grow with array size.
