#!/usr/bin/env python3
"""Score a trained checkpoint on the SAME 0 dB clips the oracle ladder used.

plan_ladder_0db.py prices what each design family could reach at 0 dB. This
measures what we actually reach, on identical clips, so the two can be divided.
That ratio -- fraction of its own ceiling that a real network achieves -- is the
only thing that turns "this family's ceiling is above target" into a statement
about whether a trained model would get there.

    /opt/anaconda3/bin/python E03/model_at_0db.py \
        --ckpt runs/g7_hop256_50k/best.pt --data ../handoff/data/edef
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

from oracle_ladder import score                      # noqa: E402
from gtcrn_lite import N_FFT                         # noqa: E402
from train_interim import Pairs, enhance, DEV        # noqa: E402
from rhear_data.manifest import read_manifest        # noqa: E402
from eval_stratified import load_model               # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--window", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    test_rows = [r for r in rows if r.get("split") == "test"]
    if len(test_rows) != len(ds):
        sys.exit("manifest test rows do not line up with the dataset")

    near0 = [i for i, r in enumerate(test_rows) if abs(float(r["snr_db"])) <= a.window]
    if len(near0) > a.n:
        sel = np.linspace(0, len(near0) - 1, a.n).astype(int)
        near0 = [near0[i] for i in sel]

    model, _ = load_model(a.ckpt)
    model.eval()
    hop = int(model.hop_.item())
    win = torch.hann_window(N_FFT, device=DEV)

    print(f"\nckpt      {a.ckpt}")
    print(f"hop       {hop}")
    print(f"clips     {len(near0)} within +/-{a.window} dB of 0 dB\n")

    acc = {m: [] for m in ("stoi", "pesq", "sdr", "sir", "sar")}
    noisy_sdr = []

    with torch.no_grad():
        for done, i in enumerate(near0):
            x, y = ds[i]
            # enhance returns (spectrum, waveform, X, spp) -- take the waveform.
            _, est_t, _, _ = enhance(model, x[None].to(DEV), win)
            est = est_t[0].cpu().numpy()
            clean, noisy = y.numpy(), x.numpy()
            o = score(clean, noisy, est)
            for m in acc:
                acc[m].append(o[m])
            noisy_sdr.append(score(clean, noisy, noisy)["sdr"])
            print(f"\r  clip {done + 1}/{len(near0)}", end="", flush=True)
    print("\n")

    res = {m: float(np.nanmean(v)) for m, v in acc.items()}
    res["dsisdr"] = res["sdr"] - float(np.nanmean(noisy_sdr))

    print(f"  STOI     {res['stoi']:.3f}")
    print(f"  PESQ     {res['pesq']:.3f}")
    print(f"  SI-SDR   {res['sdr']:.2f}")
    print(f"  dSI-SDR  {res['dsisdr']:.2f}")
    print(f"  SI-SIR   {res['sir']:.2f}")
    print(f"  SI-SAR   {res['sar']:.2f}")

    out = a.out or os.path.join(HERE, "runs", "g012", "model_at_0db.json")
    with open(out, "w") as f:
        json.dump({"ckpt": os.path.abspath(a.ckpt), "data": os.path.abspath(a.data),
                   "n_clips": len(near0), "snr_window_db": a.window,
                   "metrics": res}, f, indent=1)
    print(f"\nwritten   {out}")


if __name__ == "__main__":
    main()
