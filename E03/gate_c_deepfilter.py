#!/usr/bin/env python3
"""Gate C for the deep filter — the same learnability test, on the other lever.

The phase plan failed Gate C: both head variants beat the identity rotation by
only +0.016 / +0.019 of alignment after 1500 steps on a 16-clip MEMORISATION
task with the rest of the network frozen. The oracle says perfect phase is worth
+9.6 dB of dSI-SDR; the learnability test says this trunk cannot produce it. A
ceiling that cannot be reached is not a plan.

That points back at the deep filter, which reaches a comparable ceiling WITHOUT
predicting phase explicitly -- it applies a short complex filter across time per
bin, so the phase correction falls out of the filtering rather than having to be
regressed. df_at_0db.py measured the cascade clearing all three PS targets at
every hold length tested (PESQ 2.849-3.996, dSI-SDR 16.11-22.57).

This runs the identical protocol on the deep-filter head:

    load G7-base, freeze everything except dfdec, train on 16 clips, and ask
    whether the head can pull the output above the mask-only baseline.

The comparison that matters is against the SAME frozen trunk with the head at
its identity initialisation -- that is the mask-alone score, and the deep filter
is a pure addition to it.

    /opt/anaconda3/bin/python E03/gate_c_deepfilter.py
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

from gtcrn_lite import GTCRNLite, N_FFT                          # noqa: E402
from train_interim import Pairs, enhance, stft, loss_fn, si_sdr, DEV  # noqa: E402
from rhear_data.manifest import read_manifest                    # noqa: E402


def load_g7_into(variant, ckpt):
    sd = torch.load(ckpt, map_location=DEV)
    own = variant.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    variant.load_state_dict(own)
    return loaded, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
    ap.add_argument("--train-data", default=os.path.join(HERE, "..", "handoff", "data", "h3_20k"))
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bands", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "gate_c_deepfilter.json"))
    a = ap.parse_args()

    torch.manual_seed(0)
    win = torch.hann_window(N_FFT, device=DEV)

    m = GTCRNLite(ch=(32, 48, 48, 64), n_bands=a.bands, phase=False,
                  fullband=True, df=True, hop=256).to(DEV)
    loaded, skipped = load_g7_into(m, a.ckpt)
    print(f"\nloaded    {len(loaded)} tensors from G7-base, "
          f"{len([s for s in skipped if not s.startswith('dfdec')])} unmatched")
    for n_, p in m.named_parameters():
        p.requires_grad = n_.startswith("dfdec.")
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"trainable {trainable:,} params (deep-filter head only), "
          f"rest frozen\n")

    hdr, rows = read_manifest(os.path.join(a.train_data, "manifest.jsonl"))
    ds = Pairs(a.train_data, "train", rows, cache=False)
    xb = torch.stack([ds[i][0] for i in range(a.clips)]).to(DEV)
    yb = torch.stack([ds[i][1] for i in range(a.clips)]).to(DEV)
    S_ref = stft(yb, win, 256)

    # Baseline: the head at its identity initialisation IS the mask alone.
    m.eval()
    with torch.no_grad():
        _, wav0, _, _ = enhance(m, xb, win)
        base_sisdr = float(si_sdr(wav0, yb).mean())
        noisy_sisdr = float(si_sdr(xb, yb).mean())
    print(f"  noisy input                {noisy_sisdr:7.2f} dB")
    print(f"  mask alone (head=identity) {base_sisdr:7.2f} dB   "
          f"<- the bar the head must beat\n")

    m.train()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    hist = []
    for s in range(a.steps):
        opt.zero_grad()
        S_est, wav, X, spp = enhance(m, xb, win)
        total, _ = loss_fn(wav, yb, S_est, S_ref, spp)
        if not torch.isfinite(total):
            print("    loss went non-finite; stopping")
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in m.parameters() if p.requires_grad], 5.0)
        opt.step()
        if s % max(1, a.steps // 6) == 0 or s == a.steps - 1:
            with torch.no_grad():
                sd = float(si_sdr(wav, yb).mean())
            hist.append({"step": s, "loss": float(total), "sisdr": sd})
            print(f"    step {s:5d}   loss {float(total):9.4f}   "
                  f"SI-SDR {sd:7.2f} dB   ({sd - base_sisdr:+.2f} vs mask alone)")

    gain = hist[-1]["sisdr"] - base_sisdr if hist else float("nan")
    print(f"\n  deep-filter head gain over the mask alone   {gain:+.2f} dB")
    print(f"  (the phase head's comparable result was a +0.019 alignment gain,\n"
          f"   which the oracle prices at a fraction of a dB)")

    ok = gain > 1.0
    print(f"\n  [{'PASS' if ok else 'FAIL'}] the deep-filter head learns a "
          f"real improvement on a frozen trunk")

    with open(a.out, "w") as f:
        json.dump({"noisy_sisdr": noisy_sisdr, "mask_only_sisdr": base_sisdr,
                   "history": hist, "gain_db": gain, "pass": ok}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
