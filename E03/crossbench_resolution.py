#!/usr/bin/env python3
"""Can 300 clips even resolve the gap between us and GTCRN?

The cross-benchmark table shows GTCRN's DNS3 checkpoint at PESQ 1.643 / STOI
0.818 against our 1.635 / 0.815 -- ahead by 0.008 and 0.003. Before reading
anything into that, measure the resolution of the instrument: bootstrap our own
per-clip scores on the same 300 clips and see how wide the interval on the mean
is. If it is much wider than the gap, the two are tied and the table should say
so.

    /opt/anaconda3/bin/python E03/crossbench_resolution.py
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import N_FFT
from train_interim import Pairs, enhance, DEV
from eval_stratified import load_model
from rhear_data.manifest import read_manifest
from oracle_ladder import score

DATA = os.path.join(HERE, "..", "handoff", "data", "edef")
CKPT = os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt")
GAP = {"pesq": 0.0079, "stoi": 0.0031}      # GTCRN DNS3 minus us, from the table
BOOT = 10000

hdr, rows = read_manifest(os.path.join(DATA, "manifest.jsonl"))
ds = Pairs(DATA, "test", rows, cache=False)
model, _ = load_model(CKPT); model.eval()
win = torch.hann_window(N_FFT, device=DEV)

per = {"stoi": [], "pesq": []}
with torch.no_grad():
    for i in range(len(ds)):
        x, y = ds[i]
        _, est, _, _ = enhance(model, x[None].to(DEV), win)
        o = score(y.numpy(), x.numpy(), est[0].cpu().numpy())
        per["stoi"].append(o["stoi"]); per["pesq"].append(o["pesq"])
        if (i + 1) % 100 == 0:
            print(f"\r  {i+1}/{len(ds)}", end="", flush=True)
print("\n")

rng = np.random.default_rng(23)
print(f"G7-base on {len(ds)} E_def clips, {BOOT:,} bootstrap resamples\n")
print(f"{'metric':8s}{'mean':>9s}{'95% CI on the mean':>26s}{'half-width':>12s}"
      f"{'GTCRN gap':>11s}")
print("-" * 68)
out = {}
for k in ("pesq", "stoi"):
    v = np.asarray(per[k], float); n = len(v)
    idx = rng.integers(0, n, size=(BOOT, n))
    st = np.nanmean(v[idx], axis=1)
    lo, hi = float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))
    half = (hi - lo) / 2
    print(f"{k.upper():8s}{np.nanmean(v):9.4f}   [{lo:.4f}, {hi:.4f}]"
          f"{half:12.4f}{GAP[k]:11.4f}")
    out[k] = {"mean": float(np.nanmean(v)), "ci": [lo, hi],
              "half_width": half, "gtcrn_gap": GAP[k],
              "resolvable": bool(GAP[k] > half)}

print()
for k in ("pesq", "stoi"):
    r = out[k]
    verdict = ("the gap exceeds the noise floor" if r["resolvable"]
               else f"the gap is {r['half_width']/r['gtcrn_gap']:.0f}x SMALLER than the "
                    "interval — the two are tied")
    print(f"  {k.upper():5s} {verdict}")

json.dump(out, open(os.path.join(HERE, "runs", "g012",
                                 "crossbench_resolution.json"), "w"), indent=1)
print("\nwritten runs/g012/crossbench_resolution.json")
