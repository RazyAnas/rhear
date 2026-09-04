#!/usr/bin/env python3
"""Is the PESQ margin real, or is it sampling noise?

The fine-tuned checkpoint scores PESQ 2.524 against a 2.5 target — a margin of
0.024 over 824 utterances. A claim that thin has to survive a bootstrap before
it goes on a slide, because the same check already overturned one of this
project's headline claims earlier (the 257-bin dSI-SDR row, whose interval
crossed its target).

    /opt/anaconda3/bin/python E03/vbd_margin.py
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import N_FFT
from train_interim import enhance, DEV
from eval_stratified import load_model
from oracle_ladder import score

import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

FS = 16000
T = {"stoi": 0.85, "pesq": 2.5, "sdr": 15.0}
BOOT = 10000

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_vbdemand", "best.pt"))
ap.add_argument("--root", default="/tmp/vbd")
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "vbd_margin.json"))
a = ap.parse_args()


def find(root, key):
    for d, _, fs in os.walk(root):
        if key in d and any(f.endswith(".wav") for f in fs):
            return d


def load(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != FS:
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g).astype(np.float32)
    return x


cdir, ndir = find(a.root, "clean"), find(a.root, "noisy")
names = sorted(f for f in os.listdir(cdir)
               if f.endswith(".wav") and os.path.exists(os.path.join(ndir, f)))
model, _ = load_model(a.ckpt)
model.eval()
win = torch.hann_window(N_FFT, device=DEV)
print(f"\nckpt   {a.ckpt}\nclips  {len(names)}\nboot   {BOOT:,} resamples over clips\n")

per = {k: [] for k in ("stoi", "pesq", "sdr")}
with torch.no_grad():
    for i, nm in enumerate(names):
        c, z = load(os.path.join(cdir, nm)), load(os.path.join(ndir, nm))
        L = min(len(c), len(z)); c, z = c[:L], z[:L]
        _, est, _, _ = enhance(model, torch.from_numpy(z)[None].to(DEV), win)
        o = score(c, z, est[0].cpu().numpy())
        for k in per:
            per[k].append(o[k])
        if (i + 1) % 200 == 0:
            print(f"\r  {i+1}/{len(names)}", end="", flush=True)
print("\n")

rng = np.random.default_rng(11)
res = {}
print(f"{'metric':<10s}{'mean':>9s}{'95% CI':>22s}{'target':>9s}   verdict")
print("-" * 66)
for k, label in (("stoi", "STOI"), ("pesq", "PESQ"), ("sdr", "output SI-SDR")):
    v = np.asarray(per[k], float)
    n = len(v)
    idx = rng.integers(0, n, size=(BOOT, n))
    stat = np.nanmean(v[idx], axis=1)
    lo, hi = float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))
    mean = float(np.nanmean(v))
    t = T[k]
    verdict = ("clears" if lo > t else ("STRADDLES" if hi > t else "fails"))
    print(f"{label:<10s}{mean:9.3f}   [{lo:7.3f}, {hi:7.3f}]{t:9.2f}   {verdict}")
    res[k] = {"mean": mean, "ci": [lo, hi], "target": t, "verdict": verdict}

print("\n'clears' = the whole interval is above target, so the margin is not "
      "sampling noise.\n'STRADDLES' = 824 clips cannot settle it and the claim "
      "should be softened.")
json.dump({"ckpt": a.ckpt, "n": len(names), "bootstrap": BOOT, "result": res},
          open(a.out, "w"), indent=1)
print(f"\nwritten {a.out}")
