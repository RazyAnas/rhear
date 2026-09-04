#!/usr/bin/env python3
"""How the schematic maps onto RHEAR's three layers.

Every part on the bench sheet belongs to exactly one tier, and the reason a part
was chosen is almost always a property of the tier it serves. This draws that
correspondence so the schematic can be read as the architecture rather than as a
pile of modules.

    /opt/anaconda3/bin/python hardware/draw_architecture_map.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
TEAL, GREY, RED, INK, AMB = "#0B6E6C", "#7A8C99", "#A33B12", "#11191C", "#8A6A1F"
CARD, GHOST = "#EDF4F4", "#F4F6F7"

fig, ax = plt.subplots(figsize=(16.5, 9.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 58); ax.axis("off")


def box(x, y, w, h, label, sub=None, fc=CARD, ec=TEAL, dashed=False, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor=fc, edgecolor=ec, linewidth=1.7,
                                linestyle="--" if dashed else "-", zorder=3))
    ax.text(x + w / 2, y + h / 2 + (0.9 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=INK if not dashed else GREY,
            fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.15, sub, ha="center", va="center",
                fontsize=8.5, color=GREY, zorder=4)


def arrow(x1, y1, x2, y2, color=TEAL, label=None, dashed=False, rad=0.0, lo=0.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=1.9,
                                 linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + lo, label, ha="center",
                fontsize=8.5, color=color, zorder=5)


def lane(y, h, name, tag, color):
    ax.add_patch(FancyBboxPatch((1.2, y), 97.6, h,
                                boxstyle="round,pad=0,rounding_size=0.5",
                                facecolor="none", edgecolor=color, linewidth=1.0,
                                alpha=0.45, zorder=0))
    ax.text(2.4, y + h - 1.5, name, fontsize=12, fontweight="bold", color=color)
    ax.text(2.4, y + h - 3.4, tag, fontsize=8.5, color=GREY)


ax.text(1.2, 56.4, "The schematic, read as the architecture", fontsize=15,
        fontweight="bold", color=INK)
ax.text(1.2, 54.2, "Every part on the bench sheet belongs to one tier — and the "
        "reason it was chosen is a property of that tier.", fontsize=10, color=GREY)

# ---------------------------------------------------------------- L0
lane(33.5, 18.5, "L0   cancellation", "192 kHz · 146 µs budget", RED)
box(14, 43.5, 15, 6.0, "MIC1 · MIC3", "MAX4466, OUTSIDE each shell")
box(14, 35.0, 15, 6.0, "MIC2 · MIC4", "MAX4466, INSIDE at each ear")
box(37, 39.0, 17, 7.5, "ADAU1772", "4 ADC + 2 DAC · 38 µs", GHOST, GREY, True, 11)
box(62, 39.0, 14, 7.5, "LS1 · LS2", "16 Ω in the cups")
ax.text(84.5, 44.0, "ear", fontsize=11, color=GREY, ha="center")
arrow(29, 46.5, 37, 44.5, RED, "reference", rad=-0.12)
arrow(29, 38.0, 37, 41.0, RED, "error", rad=0.12, lo=-1.7)
arrow(54, 42.8, 62, 42.8, RED, "anti-noise")
arrow(76, 42.8, 82.5, 42.8, RED)
ax.text(52.0, 35.0, "on the 7th this loop is MEASURED, not run:\n"
        "the same two mics give coherence γ², which bounds\n"
        "what any canceller could ever achieve on this headset",
        fontsize=8.7, color=AMB, ha="center", style="italic")

# ---------------------------------------------------------------- L1
lane(15.0, 17.0, "L1   speech enhancement", "16 ms frames · 62.5 fps", TEAL)
box(14, 22.0, 15, 6.0, "MIC5", "INMP441, boom · I²S")
box(37, 22.0, 17, 6.0, "ESP32-S3", "G7 · 49,663 params")
box(62, 22.0, 14, 6.0, "MAX98357A", "I²S class-D")
box(80, 22.0, 12, 6.0, "LS1 · LS2", "same drivers")
arrow(29, 25.0, 37, 25.0, TEAL, "I²S 1")
arrow(54, 25.0, 62, 25.0, TEAL, "I²S 0")
arrow(76, 25.0, 80, 25.0, TEAL)
ax.text(45.5, 17.2, "digital microphone is fine here — its few hundred µs of "
        "internal delay is invisible inside a 16 ms frame,\n"
        "and fatal on L0. That is why the two paths use different kinds of "
        "microphone.", fontsize=8.7, color=GREY, ha="center", style="italic")

# ---------------------------------------------------------------- L2
lane(2.0, 11.5, "L2   scene engine", "1,764 params · emits coefficients, "
     "never audio", INK)
box(37, 5.5, 17, 6.0, "ESP32-S3", "scene → coefficients")
box(62, 5.5, 17, 6.0, "ADAU1772", "applies them", GHOST, GREY, True)
arrow(54, 8.5, 62, 8.5, INK, "I²C", lo=0.9)
ax.text(58, 6.9, "control, not audio", fontsize=8.5, color=INK, ha="center")
ax.text(20.5, 8.5, "no audio path", fontsize=9.5, color=INK, ha="center",
        fontweight="bold")
ax.text(20.5, 6.6, "pull the AI out mid-sentence\nand L0 keeps cancelling",
        fontsize=8.5, color=GREY, ha="center", style="italic")

ax.text(50, 0.4, "dashed = ADAU1772, on its way — the eight reserved pins are its "
        "landing site, so phase 1 adds wires rather than rework",
        fontsize=9, color=GREY, ha="center")

fig.tight_layout()
out = os.path.join(HERE, "rhear_architecture_map")
fig.savefig(out + ".svg", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"written {out}.svg\nwritten {out}.png")
