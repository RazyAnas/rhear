#!/usr/bin/env python3
"""The impact chart. Two prices is a fact; how many soldiers a budget protects
is an argument, so the chart is built on the second."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

BG, FG, DIM = "#0d1117", "#e6edf3", "#8b949e"
RED, GREEN  = "#f85149", "#3ddc84"
OUT = os.path.dirname(os.path.abspath(__file__))

TCAPS   = 180_000          # ~$2,000, US Army reference system
RHEAR   = 20_000           # our 5x-BOM estimate for a fielded unit
ARMY    = 1_475_750        # Indian Army active personnel, 2026
ARTY    = ARMY / 6         # Regiment of Artillery ~ one-sixth of Army strength
MODERN  = 185_467          # FY2026-27 capital acquisition, Rs crore

fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
fig.text(0.06, 0.945, "The price is the whole problem", color=FG,
         fontsize=29, fontweight="bold")
fig.text(0.06, 0.898,
         "Same capability. The only thing that changes is how many soldiers get one.",
         color=DIM, fontsize=14)

# ---- LEFT: soldiers protected per Rs 100 crore --------------------------
ax = fig.add_axes([0.07, 0.30, 0.38, 0.50], facecolor=BG)
n_t, n_r = int(100e7 / TCAPS), int(100e7 / RHEAR)
bars = ax.bar(["Reference system", "RHEAR"], [n_t, n_r],
              color=[RED, GREEN], width=0.5)
for b, v in zip(bars, [n_t, n_r]):
    ax.text(b.get_x() + b.get_width()/2, v + n_r*0.035, f"{v:,}",
            ha="center", color=FG, fontsize=21, fontweight="bold")
ax.set_title("Soldiers protected per ₹100 crore", color=FG, fontsize=16, pad=30)
ax.set_ylim(0, n_r * 1.30)
ax.tick_params(colors=DIM, labelsize=13)
ax.set_yticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.annotate("", xy=(1, n_r*1.16), xytext=(0, n_r*1.16),
            arrowprops=dict(arrowstyle="<->", color=DIM, lw=1.3))
ax.text(0.5, n_r*1.20, "9× more people, same money",
        ha="center", color=GREEN, fontsize=14, fontweight="bold")

# ---- RIGHT: cost to equip the artillery arm -----------------------------
ax2 = fig.add_axes([0.58, 0.38, 0.30, 0.42], facecolor=BG)
c_t, c_r = ARTY*TCAPS/1e7, ARTY*RHEAR/1e7          # Rs crore
# plotted bottom-up, so list RHEAR first to put the big bar on top
b2 = ax2.barh(["RHEAR", "Reference system"], [c_r, c_t],
              color=[GREEN, RED], height=0.45)
for b, v in zip(b2, [c_r, c_t]):
    ax2.text(v + c_t*0.03, b.get_y() + b.get_height()/2,
             f"₹{v:,.0f} cr", va="center", color=FG,
             fontsize=18, fontweight="bold")
ax2.set_title("Cost to equip the whole artillery arm\n(≈2.46 lakh personnel)",
              color=FG, fontsize=16, pad=22)
ax2.set_xlim(0, c_t*1.45)
ax2.tick_params(colors=DIM, labelsize=13)
ax2.set_xticks([])
for sp in ax2.spines.values(): sp.set_visible(False)

fig.text(0.535, 0.245,
         f"For scale: the FY2026-27 capital acquisition\n"
         f"budget is ₹{MODERN:,} crore. Equipping the\n"
         f"artillery arm would take {100*c_t/MODERN:.1f}% of it with the\n"
         f"reference system — and {100*c_r/MODERN:.2f}% with RHEAR.",
         color=GREEN, fontsize=12.5, linespacing=1.75)

fig.text(0.06, 0.045,
         "Reference system: US Army TCAPS, ~$2,000/unit, ~20,000 units fielded — its price is cited as the limit on wider deployment.\n"
         "RHEAR: ₹4,071 verified India-sourced electronics BOM as a retrofit. ₹20,000 is OUR ESTIMATE at 5× BOM for a fielded unit.\n"
         "Indian Army active strength 1,475,750 (2026); the Regiment of Artillery is approximately one-sixth of that.",
         color="#6e7681", fontsize=10, linespacing=1.75)

fig.savefig(f"{OUT}/RHEAR_impact_chart.png", facecolor=BG)
print("wrote RHEAR_impact_chart.png")
print(f"  per Rs 100 cr : {n_t:,} vs {n_r:,}  ({n_r/n_t:.0f}x)")
print(f"  artillery arm : Rs {c_t:,.0f} cr vs Rs {c_r:,.0f} cr")
print(f"  as share of modernisation budget: {100*c_t/MODERN:.1f}% vs {100*c_r/MODERN:.2f}%")
