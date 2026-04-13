"""
Wildlife Camera SoC — Physical System Plumbing Diagram
Generates project/m1/system_diagram.png

Zynq-style SoC floorplan:
  PS (left) | AXI Interconnect (center) | PL: DMA + Custom IP (right)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── Colors ─────────────────────────────────────────────────────────────────
C_PS_BG   = '#EBF5FB'
C_PS_FILL = '#2471A3'
C_PS_EC   = '#1A5276'
C_PS_LITE = '#D6E4F0'
C_DRAM_BG = '#D2B4DE'
C_DRAM_EC = '#6C3483'
C_BUS_BG  = '#D5D8DC'
C_BUS_EC  = '#566573'
C_PL_BG   = '#FEF9E7'
C_PL_EC   = '#7E5109'
C_DMA_BG  = '#FAD7A0'
C_DMA_EC  = '#CA6F1E'
C_IP_BG   = '#F9EBEA'
C_IP_EC   = '#C0392B'
C_AXS_BG  = '#FDEBD0'   # AXI-Stream port fill
C_AXL_BG  = '#EBF5FB'   # AXI-Lite port fill
C_AXL_EC  = '#5D6D7E'
C_SRAM_BG = '#D5F5E3'
C_SRAM_EC = '#1D8348'
C_XNOR_BG = '#FADBD8'
C_XNOR_EC = '#C0392B'
C_TEXT    = '#1B2631'

DATA_CLR  = '#1A5276'    # thick data path arrows
CTRL_CLR  = '#7D6608'    # thin dashed control path


def rbox(ax, x, y, w, h, label, fc, ec, lw=1.5, fontsize=10,
         bold=False, labelcolor='#1B2631', linespacing=1.5, zorder=3):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.08,rounding_size=0.12",
                           facecolor=fc, edgecolor=ec, linewidth=lw,
                           zorder=zorder)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', color=labelcolor,
            multialignment='center', linespacing=linespacing, zorder=zorder + 1)


def thick_arrow(ax, x1, y1, x2, y2, lw=3.5, color=DATA_CLR):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=20), zorder=6)


def bidir_arrow(ax, x1, y1, x2, y2, lw=3, color=DATA_CLR):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='<->', color=color, lw=lw,
                                mutation_scale=20), zorder=6)


def ctrl_seg(ax, xs, ys):
    """Draw a dashed control path through a list of (x,y) waypoints."""
    ax.plot(xs, ys, '--', color=CTRL_CLR, lw=1.6, zorder=6)


def ctrl_head(ax, x1, y1, x2, y2):
    """Arrowhead at the end of a control segment."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=CTRL_CLR, lw=1.6,
                                mutation_scale=13), zorder=7)


def ortho_data(ax, waypoints, lw=2.5, color=DATA_CLR):
    """Draw an orthogonal data path through waypoints, arrowhead at last point."""
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs, ys, '-', color=color, lw=lw, zorder=6, solid_capstyle='round')
    # arrowhead at the last segment
    ax.annotate('', xy=waypoints[-1], xytext=waypoints[-2],
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=18), zorder=7)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(10, 12.6, 'Wildlife Camera SoC — Physical System Plumbing Diagram',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C_TEXT)

# ══════════════════════════════════════════════════════════════════════════════
# REGION BACKGROUNDS
# ══════════════════════════════════════════════════════════════════════════════
ps_bg = FancyBboxPatch((0.15, 0.3), 6.9, 11.8,
                        boxstyle="round,pad=0.1,rounding_size=0.2",
                        facecolor=C_PS_BG, edgecolor=C_PS_EC,
                        linewidth=2, linestyle='--', zorder=1)
ax.add_patch(ps_bg)
ax.text(3.6, 11.88, 'Processing System (PS)',
        ha='center', va='center', fontsize=11, fontweight='bold', color=C_PS_EC)

pl_bg = FancyBboxPatch((12.5, 0.3), 7.3, 11.8,
                        boxstyle="round,pad=0.1,rounding_size=0.2",
                        facecolor=C_PL_BG, edgecolor=C_PL_EC,
                        linewidth=2, linestyle='--', zorder=1)
ax.add_patch(pl_bg)
ax.text(16.15, 11.88, 'Programmable Logic (PL)',
        ha='center', va='center', fontsize=11, fontweight='bold', color=C_PL_EC)

# Vertical dividing line
ax.plot([7.15, 7.15], [0.45, 11.6], color='#808B96', lw=2.5, linestyle=':', zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# PS BLOCKS
# ══════════════════════════════════════════════════════════════════════════════
# ARM CPU  (y: 8.8 → 10.8, center y=9.8)
rbox(ax, 0.5, 8.8, 5.9, 2.0,
     'ARM CPU Core(s)\n(Runs application SW, camera driver)',
     fc=C_PS_FILL, ec=C_PS_EC, lw=2, fontsize=10.5, bold=True, labelcolor='white')

# DDR Memory Controller  (y: 5.8 → 7.6, center y=6.7)
rbox(ax, 0.5, 5.8, 5.9, 1.8,
     'DDR Memory Controller',
     fc=C_PS_LITE, ec=C_PS_EC, lw=2, fontsize=10.5, bold=True)

# External DRAM  (y: 0.8 → 2.9, center y=1.85)
rbox(ax, 0.5, 0.8, 5.9, 2.1,
     'External DRAM (DDR4)\n(Holds full image from camera)',
     fc=C_DRAM_BG, ec=C_DRAM_EC, lw=2.5, fontsize=10.5, bold=True)
ax.text(3.4, 2.72, 'off-chip', ha='center', va='bottom', fontsize=8,
        color=C_DRAM_EC, style='italic',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_DRAM_EC, lw=1), zorder=5)

# CPU ↔ DDR Controller (bidirectional, inside PS)
bidir_arrow(ax, 3.4, 8.8, 3.4, 7.6, lw=2.5, color=C_PS_EC)

# DDR Controller ↔ External DRAM (main memory bus)
bidir_arrow(ax, 3.4, 5.8, 3.4, 2.9, lw=3.5, color=C_DRAM_EC)
ax.text(3.9, 4.35, 'Memory Bus', ha='left', va='center',
        fontsize=8, color=C_DRAM_EC, style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# AXI SYSTEM INTERCONNECT — tall central block straddling the PS/PL divide
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 7.2, 1.8, 5.1, 8.9,
     'High-Bandwidth\nAXI System\nInterconnect\n──────────\n256-bit @ 300 MHz\n(Target: 8.0 GB/s)',
     fc=C_BUS_BG, ec=C_BUS_EC, lw=2.5, fontsize=10.5, bold=True,
     labelcolor='#2C3E50')

# DDR Controller ↔ Interconnect (main data path, two lanes)
# Lane 1: DDR → Interconnect (write/read req)
thick_arrow(ax, 6.4, 7.1, 7.2, 7.1, lw=4, color=DATA_CLR)
# Lane 2: Interconnect → DDR (read data / write data)
thick_arrow(ax, 7.2, 6.7, 6.4, 6.7, lw=4, color=DATA_CLR)

# ══════════════════════════════════════════════════════════════════════════════
# PL — DMA ENGINE  (y: 8.7 → 10.6, center y=9.65)
# ══════════════════════════════════════════════════════════════════════════════
rbox(ax, 12.7, 8.7, 6.8, 1.9,
     'AXI Direct Memory Access\n(DMA) Engine',
     fc=C_DMA_BG, ec=C_DMA_EC, lw=2, fontsize=10.5, bold=True)
# DMA bottom: y=8.7   DMA left: x=12.7   DMA right: x=19.5

# DMA ↔ Interconnect: high-speed data (two lanes)
thick_arrow(ax, 12.3, 9.5, 12.7, 9.5, lw=4, color=DATA_CLR)
thick_arrow(ax, 12.7, 9.1, 12.3, 9.1, lw=4, color=DATA_CLR)
# (bus spec is embedded in the Interconnect block label)

# ══════════════════════════════════════════════════════════════════════════════
# PL — CUSTOM ACCELERATOR IP outer box  (y: 0.65 → 8.3)
# ══════════════════════════════════════════════════════════════════════════════
# Gap between IP top (8.3) and DMA bottom (8.7) = 0.4 units — used for routing
rbox(ax, 12.7, 0.65, 6.8, 7.65,
     '', fc=C_IP_BG, ec=C_IP_EC, lw=2.5, zorder=2)
ax.text(16.1, 7.78, 'Custom BinarizeConv2d Accelerator IP',
        ha='center', va='center', fontsize=10, fontweight='bold',
        color=C_IP_EC, zorder=4)

# ── IP sub-blocks ─────────────────────────────────────────────────────────
# S_AXIS — full-width input stream port  (y: 6.4 → 7.3)
rbox(ax, 13.0, 6.4, 6.2, 0.9,
     'AXI-Stream Slave (S_AXIS)  —  Stream Data Input',
     fc=C_AXS_BG, ec=C_DMA_EC, lw=1.5, fontsize=9.5, bold=True)

# AXI-Lite — right-side control port  (y: 5.1 → 6.0)
rbox(ax, 16.1, 5.1, 3.1, 0.9,
     'AXI-Lite Slave\n(S_AXI_LITE)',
     fc=C_AXL_BG, ec=C_AXL_EC, lw=1.5, fontsize=9, bold=True)

# On-Chip SRAM — left side  (y: 3.1 → 5.0)
rbox(ax, 13.0, 3.1, 2.8, 1.9,
     'On-Chip SRAM\nBuffers\n(Local Weight &\nActivation Storage)',
     fc=C_SRAM_BG, ec=C_SRAM_EC, lw=1.5, fontsize=9)

# XNOR/Popcount — right side  (y: 3.1 → 5.0)
rbox(ax, 16.1, 3.1, 3.1, 1.9,
     'XNOR / Popcount\nSpatial Compute\nArray',
     fc=C_XNOR_BG, ec=C_XNOR_EC, lw=1.5, fontsize=9, bold=True)

# M_AXIS — full-width output stream port  (y: 1.3 → 2.2)
rbox(ax, 13.0, 1.3, 6.2, 0.9,
     'AXI-Stream Master (M_AXIS)  —  Stream Data Output',
     fc=C_AXS_BG, ec=C_DMA_EC, lw=1.5, fontsize=9.5, bold=True)

# ── Internal IP data connections (orthogonal) ─────────────────────────────
# S_AXIS → SRAM  (data flows down, left column)
thick_arrow(ax, 14.4, 6.4, 14.4, 5.0, lw=2, color=C_IP_EC)

# SRAM → XNOR  (1-bit weight/activation bus, horizontal)
thick_arrow(ax, 15.8, 4.05, 16.1, 4.05, lw=2, color=C_SRAM_EC)
ax.text(15.95, 4.28, '1-bit\ndata', ha='center', va='bottom',
        fontsize=7.5, color=C_SRAM_EC, multialignment='center')

# XNOR → M_AXIS  (orthogonal: down right column → gap → left to M_AXIS)
# Route: XNOR bottom-center (17.65, 3.1) → down to y=2.5 → left to (14.4, 2.5) → down to M_AXIS top (14.4, 2.2)
ortho_data(ax, [(17.65, 3.1), (17.65, 2.5), (14.4, 2.5), (14.4, 2.2)],
           lw=2, color=C_IP_EC)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PATH: DMA ↔ Accelerator IP (thick vertical arrows)
# ══════════════════════════════════════════════════════════════════════════════
# DMA → S_AXIS (stream data in, left lane at x=14.4)
thick_arrow(ax, 14.4, 8.7, 14.4, 7.3, lw=4, color=DATA_CLR)
# label in the gap between DMA bottom and IP top
ax.text(14.8, 8.5, 'AXI-Stream In →', ha='left', va='center',
        fontsize=8, color=DATA_CLR, fontweight='bold')

# M_AXIS → DMA (stream data out, routed up the RIGHT side of the IP block)
# Route: M_AXIS right exit (19.2, 1.75) → right to x=19.65 → up to DMA center (19.65, 9.65) → left into DMA
ortho_data(ax, [(19.2, 1.75), (19.65, 1.75), (19.65, 9.65), (19.5, 9.65)],
           lw=3.5, color=DATA_CLR)
# brief label near M_AXIS exit, avoiding the right-edge clip
ax.text(19.1, 1.45, '← AXI-Stream Out', ha='right', va='top',
        fontsize=8, color=DATA_CLR, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# CONTROL PATH: CPU → Interconnect → AXI-Lite Slave (thin dashed L-shape)
# Route avoids DMA block by running through the IP-DMA gap (y=8.5),
# then descending outside all sub-blocks on the right (x=19.85)
# ══════════════════════════════════════════════════════════════════════════════
# Leg 1: CPU right → Interconnect left (horizontal at y=9.8, within CPU block)
ctrl_seg(ax, [6.4, 7.2], [9.8, 9.8])
# Leg 2: Interconnect right → down to IP-DMA gap (x=12.2, y=9.8 → y=8.5)
#   x=12.2 is LEFT of DMA left edge (12.7), so clear of DMA
ctrl_seg(ax, [12.2, 12.2], [9.8, 8.5])
# Leg 3: across the gap above IP and below DMA (y=8.5)
ctrl_seg(ax, [12.2, 19.85], [8.5, 8.5])
# Leg 4: descend on the far right, outside all sub-blocks (x=19.85)
ctrl_seg(ax, [19.85, 19.85], [8.5, 5.55])
# Leg 5: enter AXI-Lite from the right with arrowhead
ctrl_head(ax, 19.85, 5.55, 19.2, 5.55)

# Label the control path on the CPU side (above the CPU→Interconnect dashed segment)
ax.text(6.8, 9.98, 'AXI-Lite Control / Status',
        ha='left', va='bottom', fontsize=8.5, color=CTRL_CLR,
        fontweight='bold', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(facecolor=C_PS_BG,   edgecolor=C_PS_EC,   label='Processing System (PS)'),
    mpatches.Patch(facecolor=C_DRAM_BG, edgecolor=C_DRAM_EC, label='External DRAM (off-chip)'),
    mpatches.Patch(facecolor=C_BUS_BG,  edgecolor=C_BUS_EC,  label='AXI System Interconnect'),
    mpatches.Patch(facecolor=C_PL_BG,   edgecolor=C_PL_EC,   label='Programmable Logic (PL)'),
    mpatches.Patch(facecolor=C_DMA_BG,  edgecolor=C_DMA_EC,  label='AXI DMA Engine'),
    mpatches.Patch(facecolor=C_IP_BG,   edgecolor=C_IP_EC,   label='Custom Accelerator IP'),
    mpatches.Patch(facecolor=C_SRAM_BG, edgecolor=C_SRAM_EC, label='On-Chip SRAM'),
    mpatches.Patch(facecolor='white', edgecolor=DATA_CLR,
                   label='High-Speed Data Path (256-bit, thick)'),
    mpatches.Patch(facecolor='white', edgecolor=CTRL_CLR,
                   label='Control Path (AXI-Lite, thin dashed)'),
]
ax.legend(handles=legend_items, loc='lower left', fontsize=8.5,
          framealpha=0.97, bbox_to_anchor=(0.005, 0.005),
          title='Legend', title_fontsize=9.5,
          ncol=3, columnspacing=1.0, handlelength=1.5)

plt.tight_layout(pad=0.2)
plt.savefig('/Users/rebeccagilbert-croysdale/HW-4-AI-ML/project/m1/system_diagram.png',
            bbox_inches='tight', dpi=180)
print("Saved system_diagram.png")
