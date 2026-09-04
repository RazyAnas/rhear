#!/usr/bin/env python3
"""G7-base on VoiceBank+DEMAND -- the benchmark the PS's numbers come from.

PS 26052 says, verbatim: "targeting SNR > 15 dB, STOI > 0.85, and PESQ > 2.5".
It names no input-SNR condition and no test set. Every published figure those
thresholds resemble is measured on VoiceBank+DEMAND, whose test set is 824
utterances mixed at 2.5 / 7.5 / 12.5 / 17.5 dB over five unseen but benign
noises (bus, cafe, office, public square, living room).

Our own docs/02-architecture.md section 10 added "at 0 dB input SNR" to each
target and read "SNR > 15 dB" as an IMPROVEMENT of 15 dB. Both were our
choices, marked in that table as "our addition" and "disambiguates". They make
the bar strictly harder than the PS's words in three independent ways: the input
condition, the SNR reading, and the test set.

This measures G7-base on the benchmark itself, changing nothing about the model.
Reported alongside our defence-set numbers, never instead of them.

VoiceBank+DEMAND ships at 48 kHz; our model is 16 kHz, so both signals are
resampled once, together, before anything is scored.

    /opt/anaconda3/bin/python E03/eval_vbdemand.py --root /tmp/vbd
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

FS = 16000
TARGETS = {"stoi": 0.85, "pesq": 2.5, "snr_out": 15.0}

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/tmp/vbd")
ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
ap.add_argument("--n", type=int, default=0, help="0 = all 824")
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "eval_vbdemand.json"))
a = ap.parse_args()

def find_dir(root, key):
    for d, _, fs in os.walk(root):
        if key in d and any(f.endswith(".wav") for f in fs):
            return d
    return None

cdir = find_dir(a.root, "clean")
ndir = find_dir(a.root, "noisy")
if not cdir or not ndir:
    sys.exit(f"could not find clean/noisy wav dirs under {a.root}\n"
             f"  clean={cdir}  noisy={ndir}")

names = sorted(f for f in os.listdir(cdir) if f.endswith(".wav"))
names = [f for f in names if os.path.exists(os.path.join(ndir, f))]
if a.n:
    names = names[:a.n]

print(f"\nclean     {cdir}")
print(f"noisy     {ndir}")
print(f"pairs     {len(names)}")
print(f"ckpt      {a.ckpt}\n")

model, _ = load_model(a.ckpt)
model.eval()
win = torch.hann_window(N_FFT, device=DEV)

def load16(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != FS:
        from math import gcd
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g).astype(np.float32)
    return x

rows = []
with torch.no_grad():
    for i, nm in enumerate(names):
        c = load16(os.path.join(cdir, nm))
        z = load16(os.path.join(ndir, nm))
        L = min(len(c), len(z))
        c, z = c[:L], z[:L]
        _, est, _, _ = enhance(model, torch.from_numpy(z)[None].to(DEV), win)
        e = est[0].cpu().numpy()
        o = score(c, z, e)
        o["noisy_sdr"] = score(c, z, z)["sdr"]
        o["noisy_stoi"] = score(c, z, z)["stoi"]
        o["noisy_pesq"] = score(c, z, z)["pesq"]
        rows.append(o)
        if (i + 1) % 100 == 0:
            print(f"\r  {i + 1}/{len(names)}", end="", flush=True)
print()

def m(k): return float(np.nanmean([r[k] for r in rows]))

res = {"n": len(rows),
       "stoi_in": m("noisy_stoi"), "stoi_out": m("stoi"),
       "pesq_in": m("noisy_pesq"), "pesq_out": m("pesq"),
       "sisdr_in": m("noisy_sdr"), "sisdr_out": m("sdr"),
       "sir": m("sir"), "sar": m("sar")}
res["dsisdr"] = res["sisdr_out"] - res["sisdr_in"]

print(f"\nG7-base on VoiceBank+DEMAND, {res['n']} utterances")
print(f"  {'':16s}{'noisy in':>10s}{'enhanced':>10s}")
print(f"  {'STOI':16s}{res['stoi_in']:10.3f}{res['stoi_out']:10.3f}")
print(f"  {'PESQ (wb)':16s}{res['pesq_in']:10.3f}{res['pesq_out']:10.3f}")
print(f"  {'SI-SDR (dB)':16s}{res['sisdr_in']:10.2f}{res['sisdr_out']:10.2f}")
print(f"  {'SI-SIR / SI-SAR':16s}{'':10s}{res['sir']:6.2f} / {res['sar']:.2f}")

print(f"\nPS 26052, as written: SNR > 15 dB, STOI > 0.85, PESQ > 2.5")
ok = []
for label, key, t in (("STOI", "stoi_out", 0.85), ("PESQ", "pesq_out", 2.5),
                      ("output SNR (SI-SDR)", "sisdr_out", 15.0)):
    p = res[key] > t
    ok.append(p)
    print(f"  [{'MET' if p else 'not met'}] {label:22s} {res[key]:7.3f}  vs  > {t}")
print(f"\n  {sum(ok)}/3 met on the benchmark the thresholds come from.")
print(f"  (improvement reading, for completeness: dSI-SDR "
      f"{res['dsisdr']:+.2f} dB vs 15)")

res["ps_met"] = {"stoi": ok[0], "pesq": ok[1], "snr_out": ok[2]}
with open(a.out, "w") as f:
    json.dump({"ckpt": a.ckpt, "dataset": "VoiceBank+DEMAND testset",
               "targets": TARGETS, "result": res}, f, indent=1)
print(f"\nwritten   {a.out}")
