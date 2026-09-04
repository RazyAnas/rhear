#!/usr/bin/env python3
"""Evaluate against PS 26052 as the PS actually words it.

The PS describes the DATASET as covering "varying SNR levels" and then states
three targets flatly:

    SNR > 15 dB      STOI > 0.85      PESQ > 2.5

It does not name an evaluation SNR and it does not name a test set. So the
honest primary number is the whole mixed-SNR test set, not a slice of it. Our
sets span -10..+20 dB with a mean near +4.7 dB, which is exactly the condition
the PS describes.

Three cuts are reported, because two of them separate the PS's requirements from
our own choices:

  ALL              every test clip. The primary PS-aligned number.
  CLEAN CAPTURE    clips with no clipping and no preamp distortion. The PS lists
                   clipping under DATA AUGMENTATION "to improve generalization"
                   -- a training-side technique. Loading the TEST set with it as
                   well is our decision, not the PS's, and it costs us.
  BENIGN SNR       clips at 2.5 dB and above, matching the SNR range of
                   VoiceBank+DEMAND's test set, where every published PESQ that
                   these thresholds resemble is measured.

"SNR > 15 dB" is reported both ways -- as output SI-SDR and as improvement --
because the phrase is ambiguous and the panel will ask.

    /opt/anaconda3/bin/python E03/eval_ps.py --ckpt runs/g7_hop256_50k/best.pt
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

T = {"stoi": 0.85, "pesq": 2.5, "snr": 15.0}

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
ap.add_argument("--sets", default="edef,realnoise")
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "eval_ps.json"))
a = ap.parse_args()

model, _ = load_model(a.ckpt)
model.eval()
win = torch.hann_window(N_FFT, device=DEV)
print(f"\nckpt   {a.ckpt}")

report = {}
for name in a.sets.split(","):
    data = os.path.join(HERE, "..", "handoff", "data", name)
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, "test", rows, cache=False)
    test = [r for r in rows if r.get("split") == "test"]
    assert len(test) == len(ds)

    per = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
            c, z, e = y.numpy(), x.numpy(), est[0].cpu().numpy()
            o = score(c, z, e)
            o["in_sdr"] = score(c, z, z)["sdr"]
            r = test[i]
            o["clean_capture"] = (not r.get("clipping")
                                  and float(r.get("preamp_distortion_k") or 0) == 0)
            o["snr_db"] = float(r["snr_db"])
            per.append(o)
            if (i + 1) % 100 == 0:
                print(f"\r  {name} {i+1}/{len(ds)}", end="", flush=True)
    print()

    cuts = {
        "ALL — every test clip": per,
        "CLEAN CAPTURE — no clipping or distortion": [p for p in per if p["clean_capture"]],
        "BENIGN SNR — 2.5 dB and above": [p for p in per if p["snr_db"] >= 2.5],
    }
    report[name] = {}
    print(f"\n{name}")
    print(f"  {'cut':<44s}{'n':>5s}{'STOI':>8s}{'PESQ':>8s}{'out SNR':>10s}"
          f"{'ΔSNR':>8s}   met")
    print("  " + "-" * 86)
    for label, rs in cuts.items():
        if not rs:
            continue
        m = {k: float(np.nanmean([p[k] for p in rs])) for k in ("stoi", "pesq", "sdr")}
        m["dsdr"] = m["sdr"] - float(np.nanmean([p["in_sdr"] for p in rs]))
        met = sum([m["stoi"] > T["stoi"], m["pesq"] > T["pesq"], m["sdr"] > T["snr"]])
        flags = "".join("Y" if c else "." for c in
                        (m["stoi"] > T["stoi"], m["pesq"] > T["pesq"], m["sdr"] > T["snr"]))
        print(f"  {label:<44s}{len(rs):5d}{m['stoi']:8.3f}{m['pesq']:8.3f}"
              f"{m['sdr']:10.2f}{m['dsdr']:8.2f}   {met}/3  {flags}")
        report[name][label] = {"n": len(rs), **m, "met": met}

print(f"\ntargets: STOI > {T['stoi']}   PESQ > {T['pesq']}   SNR > {T['snr']} dB")
print("flags are STOI / PESQ / output-SNR in order")
json.dump({"ckpt": a.ckpt, "targets": T, "report": report}, open(a.out, "w"), indent=1)
print(f"\nwritten {a.out}")
