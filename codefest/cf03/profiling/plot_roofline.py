import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── T4 hardware ceilings ──────────────────────────────────────────────────────
PEAK_COMPUTE_GFLOPS = 8100.0   # 8.1 TFLOPS FP32
PEAK_BW_GB_S        = 320.0    # GB/s
RIDGE_POINT         = PEAK_COMPUTE_GFLOPS / PEAK_BW_GB_S  # ~25.3 FLOPs/byte

# ── measured data points ──────────────────────────────────────────────────────
kernels = [
    {"label": "Naive (16x16 block)",  "ai": 0.25,  "gflops": 1.51, "marker": "o", "color": "#e05c5c"},
    {"label": "Tiled T=8",           "ai": 2.0,   "gflops": 1.50, "marker": "s", "color": "#5c8ae0"},
]

# ── roofline x range ──────────────────────────────────────────────────────────
ai_range = np.logspace(-2, 3, 500)
roof = np.minimum(ai_range * PEAK_BW_GB_S, PEAK_COMPUTE_GFLOPS)

fig, ax = plt.subplots(figsize=(8, 5))

# roofline
ax.loglog(ai_range, roof, color="black", linewidth=2, label="Roofline (T4)")

# ridge point
ax.axvline(RIDGE_POINT, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.text(RIDGE_POINT * 1.08, 50,
        f"Ridge\n{RIDGE_POINT:.1f} FLOPs/B",
        fontsize=8, color="gray", va="bottom")

# memory and compute ceiling labels
ax.text(0.012, 0.012 * PEAK_BW_GB_S * 1.5,
        f"Mem BW\n{PEAK_BW_GB_S} GB/s",
        fontsize=8, color="black", rotation=35)
ax.axhline(PEAK_COMPUTE_GFLOPS, color="black", linestyle=":", linewidth=1, alpha=0.4)
ax.text(200, PEAK_COMPUTE_GFLOPS * 1.15,
        f"Compute ceiling\n{PEAK_COMPUTE_GFLOPS/1000:.1f} TFLOPS",
        fontsize=8, color="black", ha="center")

# data points
for k in kernels:
    ax.scatter(k["ai"], k["gflops"], marker=k["marker"], color=k["color"],
               s=120, zorder=5, label=f'{k["label"]}  ({k["gflops"]} GFLOP/s)')
    ax.annotate(
        f'  {k["label"]}\n  {k["gflops"]} GFLOP/s',
        xy=(k["ai"], k["gflops"]),
        fontsize=8, color=k["color"],
        va="center",
    )

ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=11)
ax.set_title("Roofline Model — NVIDIA T4 (N=1024 FP32 GEMM)", fontsize=12)
ax.set_xlim(1e-2, 1e3)
ax.set_ylim(1e-1, PEAK_COMPUTE_GFLOPS * 5)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
out = "gemm_roofline.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
