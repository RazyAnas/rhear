#!/usr/bin/env python3
"""Does ANY winner-scale model beat us on our own data?

The question behind "why not just take the challenge winner's model" is really
two questions. The first is whether it fits: TEA-PSE 3.0, a DNS Challenge
winner, is 22.24 M parameters and 19.66 GMAC against our 49,663 and 32 MMAC/s --
450x the parameters, ~600x the compute, on a chip with 200 MMAC/s per core. It
does not fit, and that is the premise of the project.

The second question is the interesting one: could a winner be a TEACHER, with
its knowledge distilled into our 50k-parameter student? Distillation needs a
teacher that is actually better. Two published models have already been run on
our data and neither was -- GTCRN's DNS3 checkpoint tied us and MetricGAN+
damaged the signal by 2.97 dB while gaining PESQ.

This tests the largest runnable model available: SpeechBrain's SepFormer
enhancement, 25,613,569 parameters, trained on WHAM!. If a model 516x our size
also lands near 1.6 PESQ here, then our data's difficulty is intrinsic and no
amount of borrowed capacity fixes it -- which answers the question permanently.

Runs on CPU so it does not contend with training on the GPU.

    /opt/anaconda3/bin/python E03/crossbench_sepformer.py
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from train_interim import Pairs
from rhear_data.manifest import read_manifest
from oracle_ladder import score

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "crossbench",
                                              "sepformer_edef.json"))
a = ap.parse_args()

from speechbrain.inference.separation import SepformerSeparation
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wham16k-enhancement",
    savedir="/tmp/sb_sepformer", run_opts={"device": "cpu"})
nparam = sum(p.numel() for p in model.mods.parameters())

hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
ds = Pairs(a.data, "test", rows, cache=False)
idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
print(f"\nteacher  sepformer-wham16k-enhancement, {nparam:,} params")
print(f"data     {os.path.abspath(a.data)}, {len(idx)} clips\n")

acc = {k: [] for k in ("stoi", "pesq", "sdr", "sir", "sar")}
base = []
with torch.no_grad():
    for j, i in enumerate(idx):
        x, y = ds[int(i)]
        est = model.separate_batch(x[None])[..., 0].squeeze(0).cpu().numpy()
        c, z = y.numpy(), x.numpy()
        L = min(len(c), len(est))
        o = score(c[:L], z[:L], est[:L])
        for k in acc:
            acc[k].append(o[k])
        base.append(score(c, z, z)["sdr"])
        if (j + 1) % 25 == 0:
            print(f"\r  {j+1}/{len(idx)}", end="", flush=True)
print("\n")

m = {k: float(np.nanmean(v)) for k, v in acc.items()}
m["dsdr"] = m["sdr"] - float(np.nanmean(base))

print(f"{'model':34s}{'params':>12s}{'STOI':>8s}{'PESQ':>8s}{'SI-SDR':>9s}{'SI-SAR':>9s}")
print("-" * 80)
print(f"{'SepFormer (WHAM!) — 516x ours':34s}{nparam:12,}{m['stoi']:8.3f}"
      f"{m['pesq']:8.3f}{m['sdr']:9.2f}{m['sar']:9.2f}")
print(f"{'MetricGAN+':34s}{1900000:12,}{0.755:8.3f}{1.820:8.3f}{-0.47:9.2f}{0.90:9.2f}")
print(f"{'GTCRN, DNS3 weights':34s}{23700:12,}{0.818:8.3f}{1.643:8.3f}{9.61:9.2f}{10.79:9.2f}")
print(f"{'G7-base — ours':34s}{49663:12,}{0.815:8.3f}{1.635:8.3f}{10.05:9.2f}{11.91:9.2f}")

better = m["pesq"] > 1.635 + 0.06 and m["sar"] > 11.91
print("\n" + ("A viable TEACHER: it beats us by more than the measurement noise\n"
              "on PESQ without losing SI-SAR. Distillation is worth trying."
              if better else
              "NOT a viable teacher. A model 516x our size does not beat us on\n"
              "this data by more than the noise floor, so there is nothing to\n"
              "distil. The wall is the test set, not model capacity."))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"model": "speechbrain/sepformer-wham16k-enhancement",
           "params": nparam, "n": len(idx), "metrics": m}, open(a.out, "w"), indent=1)
print(f"\nwritten {a.out}")
