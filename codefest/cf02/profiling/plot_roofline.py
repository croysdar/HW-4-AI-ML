"""
plot_roofline.py
================
Generates roofline_project.png for ECE 510 M1 deliverable.
Target platform: Apple M1
  - Peak FP32 compute : 2600 GFLOP/s
  - Peak memory BW    : 68.25 GB/s
  - Ridge point       : 2600 / 68.25 = 38.1 FLOP/byte

Run:
    python3 plot_roofline.py
Output:
    roofline_project.png  (300 dpi, saved next to this script)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Output path (same directory as this script) ──────────────────────────────
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "roofline_project.png")

# ── Platform constants ────────────────────────────────────────────────────────
PEAK_COMPUTE_GFLOPS = 2600.0    # GFLOP/s  (Apple M1 FP32, all P-cores)
PEAK_BW_GB_S        = 68.25     # GB/s     (Apple M1 unified memory)
RIDGE_AI            = PEAK_COMPUTE_GFLOPS / PEAK_BW_GB_S   # ≈ 38.1 FLOP/byte

# ── Axis limits ───────────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.1,  1000.0
Y_MIN, Y_MAX = 1.0,  10000.0

# ── Build roofline envelope ───────────────────────────────────────────────────
ai_vals = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 1000)

memory_roof  = PEAK_BW_GB_S * ai_vals          # diagonal: BW × AI
compute_roof = np.full_like(ai_vals, PEAK_COMPUTE_GFLOPS)  # horizontal ceiling
roofline     = np.minimum(memory_roof, compute_roof)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

# ── Roofline ceilings ─────────────────────────────────────────────────────────
ax.loglog(ai_vals, roofline,
          color="#2c3e50", linewidth=2.5, zorder=3,
          label=f"Roofline envelope (M1)")

# Draw the two individual ceiling lines (dashed, lighter) for context
ax.axhline(y=PEAK_COMPUTE_GFLOPS, color="#7f8c8d", linewidth=1.2,
           linestyle="--", zorder=2)
ax.loglog(ai_vals[ai_vals <= RIDGE_AI],
          PEAK_BW_GB_S * ai_vals[ai_vals <= RIDGE_AI],
          color="#7f8c8d", linewidth=1.2, linestyle="--", zorder=2)

# Ceiling labels (placed near the right edge, inside the plot)
ax.text(600, PEAK_COMPUTE_GFLOPS * 1.18,
        f"Peak Compute: {PEAK_COMPUTE_GFLOPS:,.0f} GFLOP/s",
        ha="right", va="bottom", fontsize=9, color="#2c3e50",
        fontstyle="italic")
ax.text(0.13, PEAK_BW_GB_S * 0.13 * 0.72,
        f"Mem BW: {PEAK_BW_GB_S} GB/s",
        ha="left", va="top", fontsize=9, color="#2c3e50",
        fontstyle="italic", rotation=50)

        # ── Ridge point (MOVE LABEL UP AND LEFT TO CLEAR THE HW POINT) ───────────────
ax.plot(RIDGE_AI, PEAK_COMPUTE_GFLOPS,
        marker="D", markersize=9, color="#8e44ad",
        zorder=5, label=f"Ridge Point ({RIDGE_AI:.1f}, {PEAK_COMPUTE_GFLOPS:,.0f})")
ax.annotate(f"Ridge Point\n({RIDGE_AI:.1f} FLOP/byte, {PEAK_COMPUTE_GFLOPS:,.0f} GFLOP/s)",
            xy=(RIDGE_AI, PEAK_COMPUTE_GFLOPS),
            xytext=(RIDGE_AI * 0.15, PEAK_COMPUTE_GFLOPS * 0.55), # below ceiling, left of point
            fontsize=8.5, color="#8e44ad",
            arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#8e44ad", alpha=0.85))

# ── SW Baseline point ─────────────────────────────────────────────────────────
SW_AI   = 12.34
SW_PERF = 83.0

ax.plot(SW_AI, SW_PERF,
        marker="o", markersize=11, color="#e74c3c",
        zorder=5, label=f"SW Baseline — FP32 on CPU ({SW_AI} FLOP/byte, {SW_PERF} GFLOP/s)")
ax.annotate(f"SW Baseline (FP32, CPU)\n({SW_AI} FLOP/byte, {SW_PERF} GFLOP/s)\n← Memory-Bound",
            xy=(SW_AI, SW_PERF),
            xytext=(SW_AI * 0.55, SW_PERF * 2.6),
            fontsize=8.5, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.85))

# ── Target HW Accelerator point (ADJUSTED FOR CLARITY) ───────────────────────
HW_AI   = 150.0
HW_PERF = 1200.0

ax.plot(HW_AI, HW_PERF,
        marker="^", markersize=12, color="#27ae60",
        zorder=5, label=f"Target HW — 1-bit BNN ({HW_AI} FLOP/byte, {HW_PERF:,.0f} GFLOP/s)")
ax.annotate(f"Target HW (1-bit BNN)\n({HW_AI} FLOP/byte, {HW_PERF:,.0f} GFLOP/s)\nCompute-Bound →",
            xy=(HW_AI, HW_PERF),
            xytext=(HW_AI * 0.58, HW_PERF * 0.42),
            fontsize=8.5, color="#27ae60",
            arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#27ae60", alpha=0.85))

# ── Shaded regions (memory-bound / compute-bound) ─────────────────────────────
ax.axvspan(X_MIN, RIDGE_AI, alpha=0.06, color="#3498db", zorder=1)
ax.axvspan(RIDGE_AI, X_MAX, alpha=0.06, color="#2ecc71", zorder=1)
ax.text(0.55, 2.2,  "Memory-Bound",  fontsize=9, color="#2980b9", alpha=0.7, style="italic")
ax.text(55,   2.2, "Compute-Bound", fontsize=9, color="#27ae60", alpha=0.7, style="italic")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MIN, Y_MAX)
ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12, labelpad=8)
ax.set_ylabel("Attained Performance (GFLOP/s)",   fontsize=12, labelpad=8)
ax.set_title("Roofline Model: BNN Image Classifier on Apple M1",
             fontsize=13, fontweight="bold", pad=14)

# Major and minor grid lines for log scale
ax.grid(which="major", linestyle="-",  linewidth=0.6, color="#bdc3c7", alpha=0.8)
ax.grid(which="minor", linestyle=":",  linewidth=0.4, color="#bdc3c7", alpha=0.5)
ax.minorticks_on()

# Clean tick labels (avoid scientific notation on log axes)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:g}"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, _: f"{y:g}"))

# ── Legend ────────────────────────────────────────────────────────────────────
legend = ax.legend(loc="upper left", fontsize=8.2, framealpha=0.9,
                   edgecolor="#bdc3c7", borderpad=0.8)

# ── Source annotation (bottom right) ─────────────────────────────────────────
ax.text(0.99, 0.01,
        "Platform: Apple M1  |  BW: 68.25 GB/s  |  Peak FP32: 2,600 GFLOP/s\n"
        "Source: Apple silicon spec sheet",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#7f8c8d", style="italic")

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Saved: {OUT_PATH}")
