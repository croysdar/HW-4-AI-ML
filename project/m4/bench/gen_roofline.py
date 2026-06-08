"""Generate M4 final roofline plot for BNN accelerator vs. software baseline."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))

# Axis range
ai_range = np.logspace(0, 4, 1000)  # 1 to 10000 FLOP/byte

# ── Software (Apple M5) roofline ─────────────────────────────────────────────
# M5 unified memory bandwidth: ~120 GB/s; peak CPU FP32: ~4000 GFLOP/s
m5_bw_gb   = 120.0
m5_peak_gf = 4000.0
m5_roof    = np.minimum(ai_range * m5_bw_gb, m5_peak_gf)

# ── Hardware (BNN chiplet @ 40 MHz) roofline ─────────────────────────────────
# AXI: 256-bit @ 40 MHz = 1.28 GB/s (32 bytes × 40M)
# Peak XNOR: 256 ops/cycle × 40 MHz = 10,240 GOPS
hw_bw_gb   = 1.28
hw_peak_gf = 10240.0
hw_roof    = np.minimum(ai_range * hw_bw_gb, hw_peak_gf)

# ── Plot rooflines ────────────────────────────────────────────────────────────
ax.loglog(ai_range, m5_roof, color="#4477AA", linewidth=2.0, linestyle="--",
          label="Apple M5 CPU roofline (120 GB/s, 4000 GFLOP/s)")
ax.loglog(ai_range, hw_roof, color="#EE6677", linewidth=2.0, linestyle="-",
          label="BNN chiplet roofline (1.28 GB/s AXI @ 40 MHz, 10,240 GXNOR/s)")

# Ridge points
m5_ridge = m5_peak_gf / m5_bw_gb   # ~33 FLOP/byte
hw_ridge = hw_peak_gf / hw_bw_gb   # 8000 FLOP/byte
ax.axvline(m5_ridge, color="#4477AA", linewidth=0.8, linestyle=":", alpha=0.7)
ax.axvline(hw_ridge, color="#EE6677", linewidth=0.8, linestyle=":", alpha=0.7)
ax.text(m5_ridge * 1.15, 0.5, f"M5 ridge\n{m5_ridge:.0f} F/B",
        color="#4477AA", fontsize=8, va="bottom")
ax.text(hw_ridge * 0.55, 0.5, f"HW ridge\n{hw_ridge:.0f} F/B",
        color="#EE6677", fontsize=8, va="bottom", ha="right")

# ── Operating points ─────────────────────────────────────────────────────────
# SW baseline: M5 CPU, bnn_serengeti2 (4-layer, float32)
# 151.5 FPS × 1.474 GFLOP/frame = 223 GFLOP/s; AI=47.64 FLOP/byte (conv2)
sw_ai   = 47.64
sw_perf = 223.0
ax.plot(sw_ai, sw_perf, marker="s", color="#4477AA", markersize=11, zorder=5)
ax.annotate("M5 CPU — bnn_serengeti2\n(4-layer, float32)\n223 GFLOP/s | AI=47.6 F/B",
            xy=(sw_ai, sw_perf), xytext=(3, 2000),
            fontsize=8.5, color="#4477AA",
            arrowprops=dict(arrowstyle="->", color="#4477AA", lw=1.2))

# HW measured — 4-SRAM final design (40 MHz, 2.5 FPS)
# AI: 694 MOp / 351,232 B = ~1,975 FLOP/byte (binary-packed activations)
# Attained: 2.5 FPS × 694 MOp/frame = 1.74 GOPS
hw4s_ai   = 1975.0
hw4s_perf = 1.74
ax.plot(hw4s_ai, hw4s_perf, marker="^", color="#228833", markersize=13, zorder=5)
ax.annotate("BNN chiplet\n(4-SRAM, 40 MHz — final)\n1.74 GOPS | 11.5 mW | 4.62 mJ/frame\nAI=1,975 F/B",
            xy=(hw4s_ai, hw4s_perf), xytext=(200, 20),
            fontsize=8.5, color="#228833",
            arrowprops=dict(arrowstyle="->", color="#228833", lw=1.2))

# ── Region annotations ────────────────────────────────────────────────────────
ax.fill_betweenx([0.1, hw_peak_gf * 1.5], 1, hw_ridge,
                 alpha=0.04, color="#EE6677")
ax.text(1.5, 5000, "memory-\nbound", color="#888888",
        fontsize=8, ha="left", va="top", style="italic")
ax.text(hw_ridge * 1.15, 5000, "compute-\nbound", color="#888888",
        fontsize=8, ha="left", va="top", style="italic")

# ── Labels / formatting ───────────────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
ax.set_ylabel("Attained Performance (GOPS equiv.)", fontsize=12)
ax.set_title("Roofline Analysis — BNN Accelerator vs. Software Baseline\n"
             "ECE 510 Spring 2026 | Sky130A sky130_fd_sc_hd | SW baseline: Apple M5 CPU", fontsize=12)
ax.set_xlim(1, 12000)
ax.set_ylim(0.1, hw_peak_gf * 2)
ax.grid(True, which="both", alpha=0.25, linestyle=":")

legend_patches = [
    mpatches.Patch(color="#4477AA", label="Apple M5 CPU (software baseline — bnn_serengeti2)"),
    mpatches.Patch(color="#EE6677", label="BNN chiplet roofline (256-bit AXI @ 40 MHz)"),
    mpatches.Patch(color="#228833", label="BNN chiplet — 4-SRAM final design (measured)"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

plt.tight_layout()
out = "project/m4/bench/roofline_final.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
