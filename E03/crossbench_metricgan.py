#!/usr/bin/env python3
"""A SECOND published model on our data. Is our test set the problem, or are we?

crossbench.py already ran GTCRN (ICASSP 2024, 23.7k params, published PESQ 2.87
on VoiceBank+DEMAND) over our held-out defence clips: 1.444 with its VCTK
checkpoint, 1.643 with its DNS3 checkpoint, against our 1.635. Same size, same
score.

The obvious objection is that GTCRN is tiny. So this runs MetricGAN+
(Fu et al., Interspeech 2021), which reports PESQ 3.15 on VoiceBank+DEMAND and
is ~50x larger, with the authors' own pretrained weights through SpeechBrain's
published inference path. Scored by OUR scorer so nothing differs but the model.

If a model that scores 3.15 on the benchmark also lands near 1.6 here, then no
amount of copying someone else's architecture reaches PESQ 2.5 on this data, and
the gap is the test set rather than the model.

    /opt/anaconda3/bin/python E03/crossbench_metricgan.py
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))

from train_interim import Pairs, DEV
from rhear_data.manifest import read_manifest
from oracle_ladder import score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
ap.add_argument("--window", type=float, default=2.5)
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "crossbench",
                                              "metricgan_edef.json"))
a = ap.parse_args()

from speechbrain.inference.enhancement import SpectralMaskEnhancement
model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank", savedir="/tmp/sb_mgan")

hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
ds = Pairs(a.data, "test", rows, cache=False)
test_rows = [r for r in rows if r.get("split") == "test"]
near0 = {i for i, r in enumerate(test_rows) if abs(float(r["snr_db"])) <= a.window}

print(f"\ndata      {os.path.abspath(a.data)}")
print(f"clips     {len(ds)} test, of which {len(near0)} within "
      f"+/-{a.window} dB of 0 dB\n")

allr, zerr = [], []
for i in range(len(ds)):
    x, y = ds[i]
    clean, noisy = y.numpy(), x.numpy()
    with torch.no_grad():
        est = model.enhance_batch(x[None], lengths=torch.tensor([1.0]))[0].cpu().numpy()
    o = score(clean, noisy, est)
    o["noisy_sdr"] = score(clean, noisy, noisy)["sdr"]
    allr.append(o)
    if i in near0:
        zerr.append(o)
    if (i + 1) % 50 == 0:
        print(f"\r  {i + 1}/{len(ds)}", end="", flush=True)
print()

def agg(rs, label):
    if not rs:
        return None
    d = {m: float(np.nanmean([r[m] for r in rs])) for m in ("stoi", "pesq", "sdr", "sir", "sar")}
    d["dsisdr"] = d["sdr"] - float(np.nanmean([r["noisy_sdr"] for r in rs]))
    d["n"] = len(rs)
    print(f"  {label:22s} n={d['n']:3d}  STOI {d['stoi']:.3f}  PESQ {d['pesq']:.3f}  "
          f"dSI-SDR {d['dsisdr']:+.2f}  SI-SAR {d['sar']:.2f}")
    return d

print("\nMetricGAN+ (published PESQ 3.15 on VoiceBank+DEMAND) on OUR data:")
A = agg(allr, "all test clips")
Z = agg(zerr, "0 dB +/-2.5 dB only")
print("\n  for comparison, on the same clips:")
print("    G7-base, all test          STOI 0.815  PESQ 1.635")
print("    GTCRN (DNS3 ckpt), all     STOI 0.818  PESQ 1.643")
print("    GTCRN (VCTK ckpt), all     STOI 0.765  PESQ 1.444")
print("    G7-base, 0 dB              STOI 0.724  PESQ 1.278  dSI-SDR +9.29")
print("    PS target at 0 dB          STOI 0.850  PESQ 2.500  dSI-SDR +15.00")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w") as f:
    json.dump({"model": "speechbrain/metricgan-plus-voicebank",
               "published_pesq_vbdemand": 3.15,
               "data": os.path.abspath(a.data),
               "all": A, "at_0db": Z}, f, indent=1)
print(f"\nwritten   {a.out}")
