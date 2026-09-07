#!/usr/bin/env python3
"""Is the model underfitting, or is it the data?

Listening to the A/B pack, the enhanced audio still has audible noise in it. The
question that separates the two explanations is whether the model can fit the
clips it was TRAINED on. A network that has memorised its training set and still
fails on held-out data has a generalisation problem. A network that cannot even
fit what it was shown is underfitting, and there is headroom in capacity,
schedule or optimisation rather than in data.

So: score G7-base on 300 clips drawn from its own training split, and on the 300
held-out clips, with the identical scorer.

  train ~= test   -> underfitting. The model is not memorising at all.
  train >> test   -> a generalisation gap, and more/harder data is the lever.

    /opt/anaconda3/bin/python E03/train_vs_test.py
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

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
_ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "train_vs_test.json"))
_a = _ap.parse_args()
CKPT = _a.ckpt
N = 300
model, _ = load_model(CKPT); model.eval()
print(f"ckpt {CKPT}  ({model.erb.n_bands} bands)")
win = torch.hann_window(N_FFT, device=DEV)
rng = np.random.default_rng(5)


def run(data, split, label, n=N):
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, split, rows, cache=False)
    idx = (np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
           if len(ds) > n else np.arange(len(ds)))
    acc = {k: [] for k in ("stoi", "pesq", "sdr", "sir", "sar")}
    base = []
    with torch.no_grad():
        for j, i in enumerate(idx):
            x, y = ds[int(i)]
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
            c, z = y.numpy(), x.numpy()
            o = score(c, z, est[0].cpu().numpy())
            for k in acc:
                acc[k].append(o[k])
            base.append(score(c, z, z)["sdr"])
            if (j + 1) % 100 == 0:
                print(f"\r  {label} {j+1}/{len(idx)}", end="", flush=True)
    print()
    m = {k: float(np.nanmean(v)) for k, v in acc.items()}
    m["dsdr"] = m["sdr"] - float(np.nanmean(base))
    m["n"] = len(idx)
    return m


H3 = os.path.join(HERE, "..", "handoff", "data", "h3_20k")
res = {
    "TRAIN (clips the model was fitted on)": run(H3, "train", "train"),
    "VAL (held out, same corpus)":           run(H3, "val", "val"),
    "TEST E_def (held out, harder)":         run(os.path.join(HERE, "..", "handoff", "data", "edef"), "test", "edef"),
}

print(f"\n{'cut':40s}{'n':>5s}{'STOI':>8s}{'PESQ':>8s}{'SI-SDR':>9s}"
      f"{'ΔSDR':>8s}{'SI-SAR':>9s}")
print("-" * 87)
for k, m in res.items():
    print(f"{k:40s}{m['n']:5d}{m['stoi']:8.3f}{m['pesq']:8.3f}{m['sdr']:9.2f}"
          f"{m['dsdr']:8.2f}{m['sar']:9.2f}")

tr, te = res["TRAIN (clips the model was fitted on)"], res["TEST E_def (held out, harder)"]
va = res["VAL (held out, same corpus)"]
print(f"\ntrain - val   PESQ {tr['pesq'] - va['pesq']:+.3f}   STOI {tr['stoi'] - va['stoi']:+.3f}")
print(f"train - test  PESQ {tr['pesq'] - te['pesq']:+.3f}   STOI {tr['stoi'] - te['stoi']:+.3f}")
gap = tr["pesq"] - va["pesq"]
print("\n" + ("UNDERFITTING: the model scores no better on data it was trained on\n"
               "than on data it has never seen. Capacity, schedule or optimisation\n"
               "is the limit -- not the amount of data."
               if abs(gap) < 0.10 else
               "GENERALISATION GAP: the model fits its training data much better\n"
               "than held-out data. More or harder data is the lever."))
json.dump(res, open(_a.out, "w"), indent=1)
print("\nwritten runs/g012/train_vs_test.json")
