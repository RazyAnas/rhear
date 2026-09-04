#!/usr/bin/env python3
"""Which surviving plan is actually within reach, and what does it cost to run?

plan_ladder_0db.py found which design families have a ceiling above all three PS
targets at 0 dB. A ceiling above target is necessary, not sufficient -- it says a
PERFECT model of that family would pass. This turns that into a decision by
asking, for each survivor, two questions with measured answers:

  1. What FRACTION of its own ceiling would a trained model have to reach?
     Compared against the fraction G7-base reaches of the ceiling it has today,
     on the identical clips. A plan that demands 90% where we have only ever
     managed 55% is not a plan, however high its ceiling.

  2. Does it fit the hardware? Parameter count for each band-count variant,
     against the PS's < 200 KB and the ESP32-S3's ~200 MMAC/s per core.

Reads runs/g012/plan_ladder_0db.json and runs/g012/model_at_0db.json.

    /opt/anaconda3/bin/python E03/plan_demand.py
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

METRICS = (("STOI", "stoi"), ("PESQ", "pesq"), ("dSI-SDR", "dsisdr"))

# The row whose ceiling G7-base is currently working against: our design today.
CURRENT_ROW = "R48 mask@48 bands, noisy phase"


def param_counts():
    """Parameters for each band-count variant of the current architecture."""
    import torch  # noqa: F401
    from gtcrn_lite import GTCRNLite

    out = {}
    for nb in (48, 96, 257):
        try:
            m = GTCRNLite(ch=(32, 48, 48, 64), n_bands=nb, phase=False,
                          fullband=True, df=False, hop=256)
            out[nb] = sum(p.numel() for p in m.parameters() if p.requires_grad)
        except Exception as e:                                  # noqa: BLE001
            out[nb] = f"could not build: {type(e).__name__}: {e}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=os.path.join(HERE, "runs", "g012", "plan_ladder_0db.json"))
    ap.add_argument("--model", default=os.path.join(HERE, "runs", "g012", "model_at_0db.json"))
    a = ap.parse_args()

    L = json.load(open(a.ladder))
    M = json.load(open(a.model))
    rows, targets, survivors = L["rows"], L["targets"], L["survivors"]
    cur = M["metrics"]

    if CURRENT_ROW not in rows:
        sys.exit(f"ladder has no row {CURRENT_ROW!r}")
    base_ceiling = rows[CURRENT_ROW]

    print(f"\nclips     {L['n_clips']} at 0 dB +/-{L['snr_window_db']} dB")
    print(f"model     {os.path.basename(os.path.dirname(M['ckpt']))}/"
          f"{os.path.basename(M['ckpt'])}\n")

    print("What G7-base achieves of the ceiling it has TODAY "
          f"({CURRENT_ROW}):")
    achieved = {}
    for label, m in METRICS:
        frac = cur[m] / base_ceiling[m]
        achieved[m] = frac
        print(f"  {label:8s} {cur[m]:7.3f} of {base_ceiling[m]:7.3f}   = {100 * frac:5.1f}%")

    print("\nWhat each surviving plan would DEMAND, and the gap against that:\n")
    hdr = f"{'plan'.ljust(34)}"
    for label, _ in METRICS:
        hdr += f"{label:>10s}{'gap':>8s}"
    print(hdr + "   worst gap")
    print("-" * (34 + 18 * len(METRICS) + 12))

    ranked = []
    for k in survivors:
        r = rows[k]
        line = k.ljust(34)
        gaps = []
        for label, m in METRICS:
            need = targets[m] / r[m]
            gap = need - achieved[m]          # >0 means harder than today
            gaps.append(gap)
            line += f"{100 * need:9.1f}%{100 * gap:+8.1f}"
        worst = max(gaps)
        ranked.append((worst, k, r))
        print(line + f"   {100 * worst:+6.1f} pts")

    ranked.sort()
    print("\nRanked by the hardest single demand (least demanding first):")
    for i, (worst, k, r) in enumerate(ranked, 1):
        print(f"  {i}. {k:34s} worst demand {100 * worst:+6.1f} pts "
              f"over what we manage today")

    print("\nReading it: a NEGATIVE gap means the plan asks for less of its "
          "ceiling\nthan we already extract from ours -- that metric is "
          "effectively banked.\nA positive gap is the real work.")

    print("\n\nHardware cost -- parameters by band count "
          "(ch 32/48/48/64, hop 256, fullband):")
    for nb, n in sorted(param_counts().items()):
        if isinstance(n, str):
            print(f"  {nb:3d} bands   {n}")
        else:
            print(f"  {nb:3d} bands   {n:,} params   ~{n * 4 / 1024:.0f} KB fp32   "
                  f"~{n / 1024:.0f} KB int8")
    print("\n  PS limits: model < 200 KB, <= 125 MMAC/s, RTF <= 0.75")
    print("  G7-base measured: 49,663 params, 32.34 MMAC/s of ~200 per core")


if __name__ == "__main__":
    main()
