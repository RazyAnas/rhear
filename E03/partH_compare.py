#!/usr/bin/env python3
"""Frozen baseline (synthetic-noise training) vs real-noise retrain."""
import json
b_real = json.load(open("partG_real_eval/summary.json"))["overall"]
b_syn = json.load(open("baseline_frozen/metrics.json"))["overall"]
n_real = json.load(open("runs/real/eval_real/summary.json"))["overall"]
n_syn = json.load(open("runs/real/eval_synth/summary.json"))["overall"]
print("\n  ONE VARIABLE CHANGED: training-noise domain (synthetic -> real)")
print(f"  {'':<12}{'REAL test set':>28}{'SYNTHETIC test set':>30}")
print(f"  {'metric':<12}{'frozen':>9}{'real-trained':>14}{'gain':>7}"
      f"{'frozen':>11}{'real-trained':>14}{'gain':>7}")
print("  " + "-" * 70)
for m in ("stoi", "pesq", "sisdr"):
    f_ = ".3f" if m != "sisdr" else ".2f"
    dbr = b_real[f"{m}_enhanced"] - b_real[f"{m}_noisy"]
    dnr = n_real[f"{m}_enhanced"] - n_real[f"{m}_noisy"]
    dbs = b_syn[f"{m}_enhanced"] - b_syn[f"{m}_noisy"]
    dns = n_syn[f"{m}_enhanced"] - n_syn[f"{m}_noisy"]
    print(f"  {m.upper():<12}{dbr:+9{f_}}{dnr:+14{f_}}{dnr-dbr:+7{f_}}"
          f"{dbs:+11{f_}}{dns:+14{f_}}{dns-dbs:+7{f_}}")
print("\n  (all figures are DELTA vs the noisy input on that test set)")
