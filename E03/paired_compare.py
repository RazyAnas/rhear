#!/usr/bin/env python3
"""Paired bootstrap between two checkpoints on identical clips.

The absolute CI on PESQ over 300 clips is about +/-0.06, which is wider than
most improvements this project produces. That does NOT mean the improvements
are noise: most of that width is clip-to-clip variation, and it CANCELS when
the same clips are scored by both models. The right test for "is B better than
A" is a bootstrap on the per-clip DIFFERENCE, not a comparison of two absolute
intervals.

    /opt/anaconda3/bin/python E03/paired_compare.py --a runs/g7_hop256_50k/best.pt \
        --b runs/g8_bands96/best.pt
"""
import os, sys, json, argparse, warnings
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

BOOT = 10000
ap = argparse.ArgumentParser()
ap.add_argument("--a", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
ap.add_argument("--b", default=os.path.join(HERE, "runs", "g8_bands96", "best.pt"))
ap.add_argument("--sets", default="edef,realnoise")
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "paired_g7_g8.json"))
args = ap.parse_args()

win = torch.hann_window(N_FFT, device=DEV)
ma, _ = load_model(args.a); ma.eval()
mb, _ = load_model(args.b); mb.eval()
print(f"\nA  {args.a}  ({ma.erb.n_bands} bands)")
print(f"B  {args.b}  ({mb.erb.n_bands} bands)")
print(f"paired bootstrap, {BOOT:,} resamples over clips\n")

rng = np.random.default_rng(31)
report = {}
for s in args.sets.split(","):
    data = os.path.join(HERE, "..", "handoff", "data", s)
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, "test", rows, cache=False)
    per = {m: {"a": [], "b": []} for m in ("stoi", "pesq", "sdr", "sar")}
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            xb = x[None].to(DEV)
            c, z = y.numpy(), x.numpy()
            for tag, mdl in (("a", ma), ("b", mb)):
                _, e, _, _ = enhance(mdl, xb, win)
                o = score(c, z, e[0].cpu().numpy())
                for m in per:
                    per[m][tag].append(o[m])
            if (i + 1) % 100 == 0:
                print(f"\r  {s} {i+1}/{len(ds)}", end="", flush=True)
    print()

    print(f"\n{s}")
    print(f"  {'metric':9s}{'A':>9s}{'B':>9s}{'B-A':>9s}{'95% CI on B-A':>24s}  verdict")
    print("  " + "-" * 68)
    report[s] = {}
    for m, label in (("pesq", "PESQ"), ("stoi", "STOI"),
                     ("sdr", "SI-SDR"), ("sar", "SI-SAR")):
        A = np.asarray(per[m]["a"], float)
        B = np.asarray(per[m]["b"], float)
        d = B - A                                  # paired, same clip
        n = len(d)
        idx = rng.integers(0, n, size=(BOOT, n))
        st = np.nanmean(d[idx], axis=1)
        lo, hi = float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))
        v = ("BETTER" if lo > 0 else ("worse" if hi < 0 else "not resolved"))
        print(f"  {label:9s}{np.nanmean(A):9.3f}{np.nanmean(B):9.3f}"
              f"{np.nanmean(d):+9.3f}   [{lo:+.3f}, {hi:+.3f}]      {v}")
        report[s][m] = {"a": float(np.nanmean(A)), "b": float(np.nanmean(B)),
                        "diff": float(np.nanmean(d)), "ci": [lo, hi], "verdict": v}
    print()

json.dump({"a": args.a, "b": args.b, "bootstrap": BOOT, "report": report},
          open(args.out, "w"), indent=1)
print(f"written {args.out}")
