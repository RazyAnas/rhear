#!/usr/bin/env python3
"""Prepare VoiceBank+DEMAND for fine-tuning: 48 kHz -> 16 kHz, once, to disk.

The corpus ships at 48 kHz and our model is 16 kHz. Resampling 11,572 pairs on
every epoch would dominate training time, so it is done once here.

TWO SPEAKERS ARE HELD OUT FOR VALIDATION. The 824-utterance test set is NOT
touched -- checkpoint selection happens on held-out TRAIN speakers, and the test
set is scored exactly once at the end. Selecting on the test set is how a
benchmark number stops meaning anything, and this project's whole argument this
week has been about numbers meaning what they claim.

    /opt/anaconda3/bin/python E03/prep_vbdemand.py --root /tmp/vbd
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

FS = 16000
VAL_SPEAKERS = {"p286", "p287"}      # held out from the 28 training speakers

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/tmp/vbd")
ap.add_argument("--out", default="/tmp/vbd/prep16k")
a = ap.parse_args()


def find_dir(root, *keys):
    best = None
    for d, _, fs in os.walk(root):
        if all(k in d for k in keys) and any(f.endswith(".wav") for f in fs):
            if best is None or len(d) < len(best):
                best = d
    return best


def load16(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != FS:
        g = gcd(int(sr), FS)
        x = resample_poly(x, FS // g, int(sr) // g).astype(np.float32)
    return x


cdir = find_dir(a.root, "clean_trainset")
ndir = find_dir(a.root, "noisy_trainset")
if not cdir or not ndir:
    sys.exit(f"training wavs not found under {a.root}\n  clean={cdir}\n  noisy={ndir}\n"
             "Unzip clean_trainset_28spk_wav.zip and noisy_trainset_28spk_wav.zip first.")

names = sorted(f for f in os.listdir(cdir) if f.endswith(".wav"))
names = [f for f in names if os.path.exists(os.path.join(ndir, f))]
print(f"\nclean  {cdir}\nnoisy  {ndir}\npairs  {len(names)}")

os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
os.makedirs(os.path.join(a.out, "val"), exist_ok=True)

rows = {"train": [], "val": []}
for i, nm in enumerate(names):
    spk = nm.split("_")[0]
    split = "val" if spk in VAL_SPEAKERS else "train"
    cp = os.path.join(a.out, split, nm.replace(".wav", "_clean.wav"))
    np_ = os.path.join(a.out, split, nm.replace(".wav", "_noisy.wav"))
    if not (os.path.exists(cp) and os.path.exists(np_)):
        c = load16(os.path.join(cdir, nm))
        z = load16(os.path.join(ndir, nm))
        L = min(len(c), len(z))
        sf.write(cp, c[:L], FS)
        sf.write(np_, z[:L], FS)
    rows[split].append({"id": nm[:-4], "speaker": spk,
                        "clean": os.path.relpath(cp, a.out),
                        "noisy": os.path.relpath(np_, a.out)})
    if (i + 1) % 500 == 0:
        print(f"\r  {i + 1}/{len(names)}", end="", flush=True)
print()

with open(os.path.join(a.out, "manifest.json"), "w") as f:
    json.dump({"fs": FS, "val_speakers": sorted(VAL_SPEAKERS),
               "n_train": len(rows["train"]), "n_val": len(rows["val"]),
               "rows": rows}, f)

spk = sorted({r["speaker"] for r in rows["train"]})
print(f"\ntrain  {len(rows['train'])} pairs, {len(spk)} speakers")
print(f"val    {len(rows['val'])} pairs, speakers {sorted(VAL_SPEAKERS)}")
print(f"test   untouched -- scored once, at the end")
print(f"\nwritten {a.out}/manifest.json")
