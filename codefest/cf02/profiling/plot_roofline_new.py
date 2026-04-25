"""
plot_roofline_new.py
====================
Updated roofline for ECE 510 — hybrid-precision 4-layer BNNClassifier.

Shows:
  - M1 roofline ceiling (original platform)
  - M5 roofline ceiling (current platform, specs estimated)
  - Old SW baseline   : 3-layer all-binary, measured M1 CPU
  - New SW baseline   : 4-layer hybrid, measured M5 CPU
  - HW chiplet target : 1-bit Conv2-4, XNOR engine

Run:
    python3 plot_roofline_new.py
Output:
    roofline_project_new.png  (saved next to this script)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "roofline_project_new.png")

# ── Platform constants ────────────────────────────────────────────────────────
# M1: published Apple spec sheet values.
M1_BW_GBs   = 68.25    # GB/s unified memory bandwidth
M1_PEAK_GFs = 2600.0   # GFLOP/s total SoC FP32 (CPU + GPU, all cores)
M1_RIDGE    = M1_PEAK_GFs / M1_BW_GBs   # ≈ 38.1 FLOP/byte

# M5: estimated (Apple has not published full M5 spec sheet as of Spring 2026).
# Derived from Apple Silicon generation-over-generation scaling:
#   BW:   M1=68, M2=100, M3=100, M4=120 → M5 ≈ 200 GB/s
#   Peak: M1=2.6, M4=3.8 TFLOPS SoC → M5 ≈ 5.0 TFLOPS estimated
M5_BW_GBs   = 200.0    # GB/s  (estimated)
M5_PEAK_GFs = 5000.0   # GFLOP/s (estimated)
M5_RIDGE    = M5_PEAK_GFs / M5_BW_GBs   # ≈ 25.0 FLOP/byte

# Chiplet XNOR engine
# 256-bit AXI4-Stream @ 300 MHz = 9.6 GB/s rated, 8.0 GB/s effective.
# Peak XNOR: 256 parallel XNOR units × 300 MHz × 2 ops = 153.6 GOPS
CHIPLET_BW_GBs   = 8.0
CHIPLET_PEAK_GFs = 153.6

# ── Operating points ──────────────────────────────────────────────────────────
# (label, AI FLOP/byte, Attained GFLOP/s, marker, color)
# Operating points use the DOMINANT KERNEL AI (bottleneck), not whole-model aggregate.
# Host CPU bottleneck is conv1 in both old and new models (12.34 FLOP/byte) —
# this is what limits CPU performance and correctly shows the memory-bound regime.
# Chiplet bottleneck is the lowest-AI binary layer (conv2 at hardware precision).
POINTS = [
    ("Old SW Baseline\n3-layer all-binary\nM1 CPU (measured)\nconv1 dominant kernel",
     12.34,  83.0,    "o",  "#e74c3c"),
    ("New SW Baseline\n4-layer hybrid\nM5 CPU (measured)\nconv1 still host bottleneck",
     12.34, 116.3,    "s",  "#2980b9"),
    ("HW Chiplet Target\n1-bit XNOR engine\n(Conv2-4)",
     379.1, 153.6,    "^",  "#27ae60"),
]

# ── Axis limits ───────────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.1,  2000.0
Y_MIN, Y_MAX = 1.0,  20000.0
ai_vals = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 1000)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

def roofline(ai, bw, peak):
    return np.minimum(bw * ai, peak)

# M1 roofline
r_m1 = roofline(ai_vals, M1_BW_GBs, M1_PEAK_GFs)
ax.loglog(ai_vals, r_m1, color="#e74c3c", lw=2.2, ls="-",
          label=f"M1 roofline  (BW={M1_BW_GBs} GB/s, Peak={M1_PEAK_GFs:,.0f} GFLOP/s)")

# M5 roofline
r_m5 = roofline(ai_vals, M5_BW_GBs, M5_PEAK_GFs)
ax.loglog(ai_vals, r_m5, color="#2980b9", lw=2.2, ls="-",
          label=f"M5 roofline  (BW={M5_BW_GBs} GB/s est., Peak={M5_PEAK_GFs:,.0f} GFLOP/s est.)")

# Chiplet roofline
r_chip = roofline(ai_vals, CHIPLET_BW_GBs, CHIPLET_PEAK_GFs)
ax.loglog(ai_vals, r_chip, color="#27ae60", lw=2.2, ls="-",
          label=f"Chiplet roofline  (AXI BW={CHIPLET_BW_GBs} GB/s, Peak={CHIPLET_PEAK_GFs} GFLOP/s)")

# Ridge point markers — one per platform at the knee of each roofline
for ridge, peak, color, name in [
    (M1_RIDGE,                          M1_PEAK_GFs,      "#e74c3c", "M1"),
    (M5_RIDGE,                          M5_PEAK_GFs,      "#2980b9", "M5"),
    (CHIPLET_PEAK_GFs / CHIPLET_BW_GBs, CHIPLET_PEAK_GFs, "#27ae60", "Chiplet"),
]:
    ax.scatter(ridge, peak, marker="D", s=60, color=color, zorder=6,
               edgecolors="white", linewidths=1.0)
    ax.annotate(f"{name} ridge\n{ridge:.1f} FLOP/byte",
                xy=(ridge, peak),
                xytext=(ridge * 1.35, peak * 0.55),
                fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7))


# ── Operating points ──────────────────────────────────────────────────────────
annotation_offsets = {
    "Old SW Baseline\n3-layer all-binary\nM1 CPU (measured)\nconv1 dominant kernel":      (0.4,  -60),
    "New SW Baseline\n4-layer hybrid\nM5 CPU (measured)\nconv1 still host bottleneck":    (0.4,   80),
    "HW Chiplet Target\n1-bit XNOR engine\n(Conv2-4)":                                    (-0.65, -80),
}

for label, ai, perf, marker, color in POINTS:
    ax.scatter(ai, perf, marker=marker, s=160, color=color, zorder=6,
               edgecolors="white", linewidths=1.5)
    dx, dy = annotation_offsets[label]
    xtext = ai * (1 + abs(dx)) if dx > 0 else ai / (1 + abs(dx))
    ytext = perf + dy
    ax.annotate(label,
                xy=(ai, perf), xytext=(xtext, ytext),
                fontsize=8, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.88))


# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12, labelpad=8)
ax.set_ylabel("Attained / Target Performance (GFLOP / s)", fontsize=12, labelpad=8)
ax.set_title("Roofline Model — BNN Wildlife Camera Accelerator\n"
             "M1 → M5 Platform Comparison  |  ECE 510 Spring 2026",
             fontsize=13, fontweight="bold", pad=14)

ax.grid(which="major", ls="-",  lw=0.6, color="#bdc3c7", alpha=0.8)
ax.grid(which="minor", ls=":",  lw=0.4, color="#bdc3c7", alpha=0.5)
ax.minorticks_on()

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92,
          edgecolor="#bdc3c7", borderpad=0.8)

ax.text(0.99, 0.01,
        "M5 specs estimated. Chiplet peak = 256 XNOR/cycle × 300 MHz.\n"
        "AI uses no-reuse assumption: FP32 for SW, 1-bit weights + 8-bit AXI for HW.\n"
        "Old SW measured on M1 CPU. New SW measured on M5 CPU.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#7f8c8d", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
