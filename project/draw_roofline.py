"""
draw_roofline.py
================
Generates roofline_project.png for the BNN Accelerator Chiplet project.

Shows three roofline ceilings (M1 CPU, M5 CPU, chiplet) and four
operating points:

  1. Old SW baseline  — 3-layer all-binary, measured on M1 CPU
  2. New SW baseline  — 4-layer hybrid, measured on M5 CPU
  3. HW chiplet target — 1-bit Conv2-4 after migration to chiplet XNOR engine

Run:
  python project/draw_roofline.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent / "roofline_project.png"

# ── Hardware specs ─────────────────────────────────────────────────────────────
# Memory bandwidth (GB/s) and peak CPU FP32 throughput (GFLOP/s).
# M1 numbers are from Apple spec sheet + measured baseline.
# M5 numbers are estimates (Apple has not published full M5 CPU specs);
# scale from M4 (120 GB/s BW, ~4.9 TFLOPS total SoC) to M5 generation.
# Update these if Apple publishes official M5 CPU figures.

M1_BW_GBs    = 68.25   # GB/s  unified memory bandwidth (published)
M1_PEAK_GFs  = 200.0   # GFLOP/s  peak FP32 CPU (4 P-cores, NEON, practical)

M5_BW_GBs    = 200.0   # GB/s  estimated (M4=120, M5 ~+65% based on die area)
M5_PEAK_GFs  = 400.0   # GFLOP/s  estimated

# Chiplet XNOR engine (our design target)
# 256-bit AXI4-Stream @ 300 MHz = 9.6 GB/s (rated), 8.0 GB/s effective.
# On-chip XNOR throughput: assuming 256 parallel XNOR units @ 300 MHz
CHIPLET_BW_GBs   = 8.0     # GB/s  AXI4-Stream effective
CHIPLET_PEAK_GFs = 153.6   # GFLOP/s  256 XNOR/cycle × 300 MHz × 2 (XNOR+pop)

# ── Operating points  (AI in FLOP/byte, Attained in GFLOP/s) ─────────────────
# All AI values computed analytically in draw_roofline.py / sw_baseline comments.
# Attained values are either measured or projected.

POINTS = [
    # label,                         AI,    Attained_GFs,  marker, color
    ("Old SW\n(3-layer, M1)",        46.3,  83.0,           "o",   "#4878CF"),
    ("New SW\n(4-layer hybrid, M5)", 57.9,  116.3,          "s",   "#6ACC65"),
    ("HW Chiplet\n(XNOR target)",    379.1, 153.6,          "^",   "#D65F5F"),
]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xscale("log")
ax.set_yscale("log")

ai_range = np.logspace(-1, 3.5, 500)  # 0.1 → ~3162 FLOP/byte

def roofline(ai, bw, peak):
    return np.minimum(bw * ai, peak)

# M1 roofline
r_m1 = roofline(ai_range, M1_BW_GBs, M1_PEAK_GFs)
ax.plot(ai_range, r_m1, color="#4878CF", lw=2, ls="--", label=f"M1 CPU  (BW={M1_BW_GBs} GB/s, Peak={M1_PEAK_GFs:.0f} GFLOP/s)")

# M5 roofline
r_m5 = roofline(ai_range, M5_BW_GBs, M5_PEAK_GFs)
ax.plot(ai_range, r_m5, color="#6ACC65", lw=2, ls="--", label=f"M5 CPU  (BW={M5_BW_GBs} GB/s est., Peak={M5_PEAK_GFs:.0f} GFLOP/s est.)")

# Chiplet roofline
r_chip = roofline(ai_range, CHIPLET_BW_GBs, CHIPLET_PEAK_GFs)
ax.plot(ai_range, r_chip, color="#D65F5F", lw=2, ls="--", label=f"Chiplet (BW={CHIPLET_BW_GBs} GB/s AXI, Peak={CHIPLET_PEAK_GFs:.0f} GFLOP/s)")

# Ridge points (vertical dotted lines)
for bw, peak, color in [
    (M1_BW_GBs,      M1_PEAK_GFs,      "#4878CF"),
    (M5_BW_GBs,      M5_PEAK_GFs,      "#6ACC65"),
    (CHIPLET_BW_GBs, CHIPLET_PEAK_GFs, "#D65F5F"),
]:
    ridge = peak / bw
    ax.axvline(ridge, color=color, lw=0.8, ls=":", alpha=0.5)

# Operating points
offsets = {
    "Old SW\n(3-layer, M1)":        (-0.3, -18),
    "New SW\n(4-layer hybrid, M5)": ( 0.15,  8),
    "HW Chiplet\n(XNOR target)":    ( 0.15,  8),
}
for label, ai, perf, marker, color in POINTS:
    ax.scatter(ai, perf, marker=marker, s=140, color=color, zorder=5,
               edgecolors="white", linewidths=1.2)
    dx, dy = offsets.get(label, (0.15, 8))
    ax.annotate(label, (ai, perf),
                xytext=(ai * (1 + dx) if dx > 0 else ai * (1 + dx), perf + dy),
                fontsize=8.5, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

# Memory-bound / compute-bound labels
ax.text(0.18, 14, "Memory\nBound", fontsize=8, color="gray",
        ha="center", style="italic")
ax.text(800, 30, "Compute\nBound", fontsize=8, color="gray",
        ha="center", style="italic")
ax.axvline(1, color="gray", lw=0.4, ls=":", alpha=0.3)

ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=11)
ax.set_ylabel("Attained Performance  (GFLOP / s)", fontsize=11)
ax.set_title("Roofline Analysis — BNN Wildlife Camera Accelerator\n"
             "ECE 510 Spring 2026", fontsize=12, fontweight="bold")

ax.set_xlim(0.1, 3000)
ax.set_ylim(1, 1000)
ax.grid(True, which="both", ls=":", alpha=0.3)
ax.legend(fontsize=8.5, loc="upper left")

# Annotation box
note = ("M5 CPU specs estimated.\n"
        "Chiplet peak = 256 XNOR/cycle × 300 MHz.\n"
        "AI computed with no-reuse assumption (float32 SW,\n"
        "1-bit weights + 8-bit AXI for HW chiplet).")
ax.text(0.99, 0.03, note, transform=ax.transAxes,
        fontsize=7, color="gray", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
