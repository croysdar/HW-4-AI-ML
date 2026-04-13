"""
BNN Accelerator Chiplet — High-Level Block Diagram (Week 1 draft)
Generates algorithm_diagram.png

Five elements per the week 1 checklist:
  host | interface | chiplet boundary | compute engine | on-chip memory
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor('white')


def rbox(ax, x, y, w, h, label, fc='#D6E4F0', ec='#2C3E50',
         lw=1.5, fontsize=11, bold=False):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1,rounding_size=0.2",
                           facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color='#1B2631',
            multialignment='center', linespacing=1.6)


def arr(ax, x1, y1, x2, y2, color='#1A5276', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


# ─────────────────────────────────────────────────────────────────────────────
# Title — sits entirely above the chiplet boundary
# ─────────────────────────────────────────────────────────────────────────────
ax.text(8, 8.65, 'BNN Accelerator Chiplet — High-Level Block Diagram',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#1B2631')

# ─────────────────────────────────────────────────────────────────────────────
# Host block  (right edge: x = 3.2)
# SW partition per partition_rationale.md §2:
#   data loading, raw input binarization, BatchNorm2d, pooling, linear classifier
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, 0.2, 2.0, 3.0, 4.0,
     'Host CPU\n(Data Load, Binarization,\nBatchNorm2d,\nPooling, Linear)',
     fc='#D6E4F0', ec='#2980B9', lw=2, fontsize=10)
# vertical center: y = 4.0

# ─────────────────────────────────────────────────────────────────────────────
# Interface corridor  (x: 3.2 → 5.4, 2.2 units wide)
# Protocol: AXI4-Stream 256-bit @ 300 MHz  (interface_selection.md §1)
# ─────────────────────────────────────────────────────────────────────────────
# Top arrow  — input path (host → chiplet)
arr(ax, 3.2, 4.55, 5.4, 4.55, color='#1A5276', lw=2.5)
# Bottom arrow — output path (chiplet → host)
arr(ax, 5.4, 3.45, 3.2, 3.45, color='#5D6D7E', lw=2.0)

# Data-flow labels, clear of each arrow and of the protocol text
ax.text(4.3, 4.78, '1-bit Input Tensors',
        ha='center', va='bottom', fontsize=9,
        color='#1A5276', fontweight='bold')
ax.text(4.3, 3.22, 'Output Activations',
        ha='center', va='top', fontsize=9,
        color='#5D6D7E', fontweight='bold')

# Protocol name — centered between the two arrows
ax.text(4.3, 4.0,
        'AXI4-Stream\n(256-bit @ 300 MHz)',
        ha='center', va='center', fontsize=9,
        color='#2C3E50', style='italic', multialignment='center')

# ─────────────────────────────────────────────────────────────────────────────
# Chiplet boundary  (left edge at x = 5.3, top at y = 8.0 → below title)
# ─────────────────────────────────────────────────────────────────────────────
chiplet = FancyBboxPatch((5.3, 0.4), 10.4, 7.6,
                          boxstyle="round,pad=0.15,rounding_size=0.3",
                          facecolor='#F8F9FA', edgecolor='#1B7A34', linewidth=2.5)
ax.add_patch(chiplet)
ax.text(10.5, 7.78, 'Chiplet Boundary',
        ha='center', va='center', fontsize=10.5, color='#1B7A34', fontweight='bold')

# Short internal continuation (chiplet entry → memory block)
arr(ax, 5.4, 4.55, 6.1, 4.55, color='#1A5276', lw=2.0)

# ─────────────────────────────────────────────────────────────────────────────
# On-chip SRAM  (right edge: x = 9.8)
# Stores 1-bit weights AND binarized activations  (partition_rationale.md §2)
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, 6.1, 2.0, 3.7, 4.0,
     'On-Chip SRAM\n(1-bit weights &\nactivations)',
     fc='#D5F5E3', ec='#1D8348', lw=1.8, fontsize=11)
# center: (7.95, 4.0)   right edge: x = 9.8

# ─────────────────────────────────────────────────────────────────────────────
# Weight bus: On-Chip SRAM → BinarizeConv2d Engine
# ─────────────────────────────────────────────────────────────────────────────
arr(ax, 9.8, 4.0, 10.6, 4.0, color='#1D8348', lw=2.0)
ax.text(10.2, 4.22, '1-bit weights',
        ha='center', va='bottom', fontsize=9,
        color='#1D8348', fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# BinarizeConv2d Engine  (right edge: x = 15.3)
# All 3 BinarizeConv2d layers  (partition_rationale.md §2)
# XNOR replaces FP32 MACs; Popcount replaces accumulation
# ─────────────────────────────────────────────────────────────────────────────
rbox(ax, 10.6, 2.0, 4.7, 4.0,
     'BinarizeConv2d Engine\n(3× Conv Layers)\nXNOR + Popcount\nspatial logic',
     fc='#FADBD8', ec='#C0392B', lw=1.8, fontsize=11, bold=True)
# center: (12.95, 4.0)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color='#D6E4F0', label='Host (SW)'),
    mpatches.Patch(color='#D5F5E3', label='On-chip SRAM'),
    mpatches.Patch(color='#FADBD8', label='Compute engine (HW)'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=9,
          framealpha=0.9, bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.3)
plt.savefig('/Users/rebeccagilbert-croysdale/HW-4-AI-ML/project/algorithm_diagram.png',
            bbox_inches='tight', dpi=180)
print("Saved algorithm_diagram.png")
