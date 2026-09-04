#!/usr/bin/env python3
"""Where every part physically goes on the 3M Peltor X3A.

The schematic says what connects to what. This says what is taped where, and it
is drawn as a cross-section through both cups because the one distinction that
matters -- OUTSIDE the shell versus INSIDE at the ear -- is invisible in a
photograph or a front view.

Part references match rhear_phase0_netlist.py exactly, so a reference here can
be looked up there and vice versa.

    /opt/anaconda3/bin/python hardware/draw_headset.py
"""
import os, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
NET = json.load(open(os.path.join(HERE, "rhear_phase0_netlist.json")))

TEAL, GREY, RED, INK, AMB = "#0B6E6C", "#7A8C99", "#A33B12", "#11191C", "#8A6A1F"
SHELL, CAVITY, HEAD = "#DCE6E6", "#F3F7F7", "#E8E2D8"

fig, ax = plt.subplots(figsize=(15.5, 9.2))
ax.set_xlim(-11.5, 11.5)
ax.set_ylim(-8.6, 6.4)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, fc, ec=INK, lw=1.4, r=0.12, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def txt(x, y, s, size=9.5, color=INK, ha="center", va="center", weight="normal", z=6):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
            fontweight=weight, zorder=z, family="DejaVu Sans")


def mic(x, y, ref, z=7):
    ax.add_patch(Circle((x, y), 0.30, facecolor="white", edgecolor=TEAL,
                        linewidth=2.0, zorder=z))
    txt(x, y, ref.replace("MIC", ""), 8.5, TEAL, weight="bold", z=z + 1)


def lead(x1, y1, x2, y2, color=TEAL, ls="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=1.3, ls=ls, zorder=3)


# ------------------------------------------------------------------ the head
box(-1.7, -2.6, 3.4, 5.2, HEAD, GREY, 1.2, 0.5)
txt(0, 1.9, "head", 10, GREY)
txt(0, -0.2, "top view\ncross-section", 8.5, GREY)
for sx in (-1, 1):
    ax.plot([sx * 1.7, sx * 1.7], [-0.8, 0.8], color=GREY, lw=2.6, zorder=2)
    txt(sx * 1.32, -1.7, "ear", 8, GREY)

# ------------------------------------------------------------------ the cups
for side, sx, cup, ref_out, ref_in, ls in (
        ("LEFT", -1, "left", "MIC1", "MIC2", "LS1"),
        ("RIGHT", 1, "right", "MIC3", "MIC4", "LS2")):
    x_shell_out = sx * 7.4
    x_shell_in = sx * 6.6
    x_cush = sx * 2.1

    # plastic shell
    box(min(x_shell_out, x_shell_in), -2.9, 0.8, 5.8, SHELL, INK, 1.6, 0.10, z=2)
    # sealed cavity between shell and cushion
    x0, w = (x_shell_in, abs(x_cush - x_shell_in)) if sx < 0 else (x_cush, abs(x_shell_in - x_cush))
    box(x0, -2.5, w, 5.0, CAVITY, GREY, 1.1, 0.10, z=1)
    # cushion sealing on the head
    box(x_cush - (0.5 if sx > 0 else 0.0), -2.2, 0.5, 4.4, "#CBD8D8", GREY, 1.0, 0.16, z=2)

    txt(sx * 5.0, 3.4, f"{side} cup", 11, INK, weight="bold")
    txt(sx * 5.0, 2.95, "3M Peltor X3A", 8.5, GREY)

    # reference microphone, on the OUTER face of the shell
    mx = x_shell_out + sx * 0.62
    mic(mx, 1.5, ref_out)
    txt(sx * 9.9, 1.5, f"{ref_out}  MAX4466\nREFERENCE\ntaped on the OUTSIDE",
        8.5, TEAL, ha="right" if sx < 0 else "left")
    lead(mx, 1.5, x_shell_out, 1.5)

    # error microphone, INSIDE the cavity at the ear
    ex = x_cush - sx * -0.85 if False else (x_cush + sx * 0.85 * -1)
    ex = x_cush + (0.85 if sx < 0 else -0.85)
    mic(ex, 1.5, ref_in)
    # label sits over the cavity, not over the head
    txt(ex + (-1.15 if sx < 0 else 1.15), 2.45,
        f"{ref_in}  ERROR\ninside, at the ear", 8.5, TEAL)

    # 7 cm, through the shell -- the number the causality budget comes from
    ax.annotate("", xy=(mx, 0.55), xytext=(ex, 0.55),
                arrowprops=dict(arrowstyle="<->", color=AMB, lw=1.4))
    txt((mx + ex) / 2, 0.18, "7 cm", 9, AMB, weight="bold")

    # driver, in the cavity
    ax.add_patch(Rectangle((sx * 4.9 - 0.45, -1.9), 0.9, 1.4, facecolor="white",
                           edgecolor=INK, lw=1.5, zorder=4))
    txt(sx * 4.9, -1.2, ls, 8.5, INK, weight="bold", z=5)
    txt(sx * 4.9, -2.35, "16 Ω\n35 mm", 8, GREY)

    # the error mic's wire leaves under the cushion -- no drilling
    ax.plot([ex, ex, x_cush + (0.1 if sx < 0 else -0.1)],
            [1.5, -2.55, -2.55], color=TEAL, lw=1.3, zorder=3)

txt(0, -3.25, "the error mics' wires thread UNDER the ear cushion — do not drill the cups",
    9, AMB, weight="bold")

# ------------------------------------------------------------------ boom mic
ax.add_patch(FancyArrowPatch((-6.6, -2.9), (-3.4, -5.0), connectionstyle="arc3,rad=0.28",
                             arrowstyle="-", color=GREY, lw=2.4, zorder=2))
mic(-3.2, -5.1, "MIC5")
txt(-2.7, -5.1, "MIC5  INMP441 — boom, in front of the mouth\n"
                "I²S, digital. Voice pickup only, never on the ANC path.",
    8.5, TEAL, ha="left")

# ------------------------------------------------------------------ the board
box(-2.6, -8.2, 5.2, 1.9, "white", INK, 1.6, 0.14, z=3)
txt(0, -7.0, "ESP32-S3 + MAX98357A", 10, INK, weight="bold", z=4)
txt(0, -7.65, "830-tie breadboard  ·  powerbank over USB", 8.5, GREY, z=4)

for x0, y0 in ((-8.02, 1.5), (-2.0, -2.55), (-4.9, -1.9),
               (8.02, 1.5), (2.0, -2.55), (4.9, -1.9), (-3.2, -5.4)):
    ax.plot([x0, x0 * 0.35, 0], [y0, -6.3, -6.3], color=GREY, lw=1.0,
            ls=(0, (4, 3)), zorder=1)

txt(0, -8.55, "every microphone and driver lead runs back to the board — "
              "seven cables, all on headers so nothing is re-soldered in phase 1",
    9, GREY)

# ------------------------------------------------------------------ legend
txt(-11.2, 5.9, "RHEAR Phase 0 — what goes where on the headset", 13, INK,
    ha="left", weight="bold")
txt(-11.2, 5.3, "Four analog microphones (MAX4466): a REFERENCE outside each shell and an "
                "ERROR inside each cup at the ear.", 9.5, GREY, ha="left")
txt(-11.2, 4.9, "That pairing is the feed-forward ANC topology, and it is what makes the "
                "coherence ceiling a per-cup result.", 9.5, GREY, ha="left")

fig.tight_layout()
out = os.path.join(HERE, "rhear_headset_placement")
fig.savefig(out + ".svg", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"written {out}.svg\nwritten {out}.png")
