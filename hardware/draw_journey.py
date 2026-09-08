#!/usr/bin/env python3
"""Ideation round to final round, on the same 300 held-out clips throughout.

Every model number here is measured on the identical E_def test set (noisy STOI
0.769, PESQ 1.291, SI-SDR 2.50 dB), so the stages are directly comparable to
each other -- which is the only way a progress chart means anything.

    /opt/anaconda3/bin/python hardware/draw_journey.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAL, GREY, RED, INK, AMB = "#0B6E6C", "#7A8C99", "#A33B12", "#11191C", "#8A6A1F"
SOFT = "#D8EAE9"

#           stage        STOI   PESQ  outSNR  MMAC/s  params  note
ROWS = [("H3",          0.788, 1.529,  9.59,  33.6, 22988, "first real-noise model"),
        ("G0",          0.810, 1.604,  9.92,  33.2, 22956, "phase branch removed"),
        ("G1",          0.792, 1.493,  8.89,  33.2, 22956, "perceptual loss — rejected"),
        ("G2",          0.814, 1.629, 10.05,  37.1, 24975, "full-band branch"),
        ("G5",          0.813, 1.617, 10.07,  37.1, 24975, "mild perceptual — wash"),
        ("G6",          0.816, 1.642, 10.15,  56.4, 26479, "deep filter"),
        ("G7",          0.815, 1.635, 10.05,  20.9, 49663, "hop 256 — 2x params, 2.7x less compute")]
NOISY = (0.769, 1.291, 2.50)

names = [r[0] for r in ROWS]
x = np.arange(len(ROWS))
rej = {2}                       # G1 was rejected; draw it hollow

fig = plt.figure(figsize=(16.5, 10.4))
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22,
                      left=0.055, right=0.985, top=0.855, bottom=0.075)

fig.text(0.055, 0.960, "Ideation round → final round", fontsize=18,
         fontweight="bold", color=INK)
fig.text(0.055, 0.925, "Every model measured on the same 300 held-out defence "
         "clips, so the stages compare to each other. The benchmark panel is a "
         "different corpus and is labelled as one.",
         fontsize=10.5, color=GREY)


def bars(ax, vals, base, title, ylab, fmt="{:.3f}", pad=0.012):
    cols = [SOFT if i in rej else TEAL for i in range(len(vals))]
    edge = [RED if i in rej else TEAL for i in range(len(vals))]
    ax.bar(x, vals, color=cols, edgecolor=edge, linewidth=1.6, zorder=3)
    ax.axhline(base, color=GREY, ls="--", lw=1.3, zorder=2)
    ax.text(len(vals) - 0.4, base, f"  noisy input {fmt.format(base)}",
            fontsize=9, color=GREY, va="bottom", ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + pad, fmt.format(v), ha="center", fontsize=9.5,
                color=RED if i in rej else INK,
                fontweight="bold" if i == len(vals) - 1 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
    ax.set_ylabel(ylab, fontsize=10, color=GREY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)


ax = fig.add_subplot(gs[0, 0])
bars(ax, [r[2] for r in ROWS], NOISY[1], "Perceptual quality  ·  PESQ", "PESQ (wb)")
ax.set_ylim(1.15, 1.78)

ax = fig.add_subplot(gs[0, 1])
bars(ax, [r[1] for r in ROWS], NOISY[0], "Intelligibility  ·  STOI", "STOI", pad=0.0035)
ax.set_ylim(0.755, 0.838)

# ---- compute vs capacity: the G7 story
ax = fig.add_subplot(gs[1, 0])
mm = [r[4] for r in ROWS]; pp = [r[5] / 1000 for r in ROWS]
ax.bar(x, mm, color=[SOFT if i in rej else TEAL for i in range(len(mm))],
       edgecolor=TEAL, linewidth=1.6, zorder=3, label="compute (MMAC/s)")
for i, v in enumerate(mm):
    ax.text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=9.5, color=INK,
            fontweight="bold" if i == len(mm) - 1 else "normal")
ax2 = ax.twinx()
ax2.plot(x, pp, color=AMB, marker="o", lw=2.0, ms=6, zorder=4, label="parameters (k)")
for i, v in enumerate(pp):
    ax2.text(i, v - 3.4, f"{v:.1f}k", ha="center", fontsize=9, color=AMB)
ax2.set_ylabel("parameters (thousands)", fontsize=10, color=AMB)
ax2.set_ylim(15, 62); ax2.spines[["top"]].set_visible(False)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
ax.set_ylabel("MMAC/s", fontsize=10, color=GREY)
ax.set_ylim(0, 68)
ax.set_title("Cost on the chip  ·  the move that mattered", fontsize=13,
             fontweight="bold", color=INK, loc="left", pad=10)
ax.spines[["top"]].set_visible(False); ax.grid(axis="y", alpha=0.25, zorder=0)
ax.annotate("G7: twice the parameters,\n2.7× less compute — the network\n"
            "runs 4× less often (hop 64 → 256)",
            xy=(6, 22.5), xytext=(0.55, 58), fontsize=9.5, color=AMB,
            arrowprops=dict(arrowstyle="->", color=AMB, lw=1.5,
                            connectionstyle="arc3,rad=-0.18"))

# ---- benchmark panel
ax = fig.add_subplot(gs[1, 1])
labels = ["STOI\n(×100)", "PESQ\n(×20)", "output SNR\n(dB)"]
noisy_b = [92.1, 39.3, 8.45]
zero    = [93.1, 47.1, 17.71]
tuned   = [93.6, 50.6, 18.34]
tgt     = [85.0, 50.0, 15.0]
w = 0.26
xb = np.arange(3)
ax.bar(xb - w, noisy_b, w, color="#E4EBEB", edgecolor=GREY, lw=1.3, label="noisy input", zorder=3)
ax.bar(xb,      zero,   w, color=SOFT, edgecolor=TEAL, lw=1.6, label="G7-base, zero-shot", zorder=3)
ax.bar(xb + w,  tuned,  w, color=TEAL, edgecolor=TEAL, lw=1.6, label="fine-tuned", zorder=3)
for i, t in enumerate(tgt):
    ax.plot([i - 1.6 * w, i + 1.6 * w], [t, t], color=RED, lw=2.2, zorder=5)
ax.text(0 - 1.62 * w, tgt[0] + 2.4, "PS target", fontsize=9.5, color=RED, ha="left")
for i, (z, t2) in enumerate(zip(zero, tuned)):
    ax.text(i, z + 1.4, f"{z:.1f}", ha="center", fontsize=9, color=GREY)
    ax.text(i + w, t2 + 1.4, f"{t2:.1f}", ha="center", fontsize=9.5,
            color=INK, fontweight="bold")
# the one interval that matters
ax.errorbar(1 + w, 50.6, yerr=[[50.6 - 49.74], [51.40 - 50.6]], color=RED,
            capsize=7, lw=2.4, zorder=7, fmt="none")
ax.plot([1 + w], [50.6], marker="o", ms=5, color=RED, zorder=8)
ax.text(1 + w + 0.17, 56.0, "95% CI crosses\nthe target — PESQ is\nreached, not beaten",
        fontsize=8.8, color=RED)
ax.set_xticks(xb); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylim(0, 108)
ax.set_title("Against the PS targets  ·  VoiceBank+DEMAND, 824 clips",
             fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)
ax.legend(fontsize=9.5, frameon=False, loc="upper center", ncol=3,
          bbox_to_anchor=(0.5, -0.12), handlelength=1.4, columnspacing=2.0)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.25, zorder=0)
ax.text(0.5, -0.235, "STOI and PESQ rescaled onto one axis; the red bars are the "
        "targets at their rescaled height.", transform=ax.transAxes,
        fontsize=8.5, color=GREY, ha="center")

out = os.path.join(HERE, "rhear_journey")
fig.savefig(out + ".svg", bbox_inches="tight")
fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"written {out}.svg\nwritten {out}.png")
