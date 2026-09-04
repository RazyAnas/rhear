#!/usr/bin/env python3
"""Resolve the deep-filter oracle's contradiction, at 0 dB.

The repo holds two irreconcilable numbers for the SAME row and the SAME hold
window (24 frames):

    runs/g012/oracle_w24.json    H deep filter <5k + ERB   PESQ 3.297
    runs/g012/oracle_ladder.json H deep filter <5k + ERB   PESQ 2.275

Clip sampling cannot explain a gap that size (their "A noisy" rows differ by
0.013 PESQ). One of the two was produced by an earlier version of the cascade --
oracle_ladder.py's own docstring records that an earlier H had the deep filter
REPLACING the mask below 5 kHz instead of refining it, "a different (and worse)
architecture". Whichever way round it is, the source comment in
model/gtcrn_lite.py quotes 3.297 as the deep filter's ceiling, and a plan that
retires the deep-filter direction cannot rest on a number that is in dispute.

This re-measures rows B / G / Gb / H with the CURRENT code, on the 0 dB clips,
across the hold window the result is known to be sensitive to.

    /opt/anaconda3/bin/python E03/df_at_0db.py
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

from oracle_ladder import band_limit, score, deep_filter_oracle, istft_np  # noqa: E402
from gtcrn_lite import N_FFT, FS                                          # noqa: E402
from train_interim import Pairs, stft, istft, DEV                         # noqa: E402
from rhear_data.manifest import read_manifest                             # noqa: E402

TARGETS = {"stoi": 0.85, "pesq": 2.5, "dsisdr": 15.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--window", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--taps", type=int, default=5)
    ap.add_argument("--holds", default="8,24,48",
                    help="DF coefficient hold length in frames -- the knob the "
                         "result is sensitive to")
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "df_at_0db.json"))
    a = ap.parse_args()

    holds = [int(h) for h in a.holds.split(",")]

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    test_rows = [r for r in rows if r.get("split") == "test"]
    near0 = [i for i, r in enumerate(test_rows) if abs(float(r["snr_db"])) <= a.window]
    if len(near0) > a.n:
        sel = np.linspace(0, len(near0) - 1, a.n).astype(int)
        near0 = [near0[i] for i in sel]

    names = ["A noisy", "B ideal@48, noisy phase"]
    for h in holds:
        names += [f"G deep filter, all bins (hold {h})",
                  f"Gb bounded (hold {h})",
                  f"H deep filter <5k + ERB (hold {h})"]

    acc = {k: {m: [] for m in ("stoi", "pesq", "sdr")} for k in names}
    win = torch.hann_window(N_FFT, device=DEV)
    bin5k = int(round(5000 / (FS / N_FFT)))

    print(f"\ndata      {os.path.abspath(a.data)}")
    print(f"clips     {len(near0)} at 0 dB +/-{a.window} dB")
    print(f"taps      {a.taps}   holds {holds}   5 kHz = bin {bin5k}\n")

    for done, i in enumerate(near0):
        x, y = ds[i]
        n = x.shape[-1]
        Xt = stft(x[None].to(DEV), win)[0]
        St = stft(y[None].to(DEV), win)[0]
        clean, noisy = y.numpy(), x.numpy()

        mag_x, mag_s = Xt.abs(), St.abs()
        ph_x = Xt.angle()
        irm = torch.clamp(mag_s / (mag_x + 1e-8), max=1.0)
        m48 = band_limit(irm[None], 48)[0]

        out = {
            "A noisy": noisy,
            "B ideal@48, noisy phase":
                istft(((m48 * mag_x) * torch.exp(1j * ph_x))[None], win, n)[0].cpu().numpy(),
        }

        Xn, Sn = Xt.cpu().numpy(), St.cpu().numpy()
        Ym = Xn * m48.cpu().numpy()

        for h in holds:
            Yg = deep_filter_oracle(Xn, Sn, N=a.taps, window=h)
            out[f"G deep filter, all bins (hold {h})"] = istft_np(Yg, n)

            sc = np.minimum(1.0, np.abs(Xn) / (np.abs(Yg) + 1e-12))
            out[f"Gb bounded (hold {h})"] = istft_np(Yg * sc, n)

            # H is the CASCADE: ERB gains first everywhere, then the deep filter
            # REFINES the already-masked low band. Fitting stage 2 on the masked
            # signal is what lets it act as identity where the mask was right.
            Yr = deep_filter_oracle(Ym, Sn, N=a.taps, fmax_bin=bin5k, window=h)
            Yh = Ym.copy()
            Yh[:bin5k] = Yr[:bin5k]
            sh = np.minimum(1.0, np.abs(Xn) / (np.abs(Yh) + 1e-12))
            out[f"H deep filter <5k + ERB (hold {h})"] = istft_np(Yh * sh, n)

        for k, est in out.items():
            o = score(clean, noisy, est)
            for m in acc[k]:
                acc[k][m].append(o[m])
        print(f"\r  clip {done + 1}/{len(near0)}", end="", flush=True)
    print("\n")

    res = {k: {m: float(np.nanmean(v)) for m, v in d.items()} for k, d in acc.items()}
    base = res["A noisy"]["sdr"]
    for k in res:
        res[k]["dsisdr"] = res[k]["sdr"] - base

    w = max(len(k) for k in names)
    print(f"{'row'.ljust(w)}   STOI     PESQ   dSI-SDR   clears all three?")
    print("-" * (w + 44))
    for k in names:
        r = res[k]
        miss = [n for n, m in (("STOI", "stoi"), ("PESQ", "pesq"), ("dSI-SDR", "dsisdr"))
                if r[m] < TARGETS[m]]
        v = "YES" if not miss else "no  (" + ", ".join(miss) + ")"
        print(f"{k.ljust(w)}  {r['stoi']:.3f}   {r['pesq']:6.3f}   {r['dsisdr']:6.2f}   {v}")

    print("\nThe hold length is not a free parameter: it is how many frames one "
          "set of\ncoefficients is held across. A SHORT hold means the oracle "
          "re-solves more often,\nwhich a real per-frame predictor would have "
          "to match. Read the LONGEST hold as\nthe conservative number.")

    with open(a.out, "w") as f:
        json.dump({"data": os.path.abspath(a.data), "n_clips": len(near0),
                   "taps": a.taps, "holds": holds, "targets": TARGETS,
                   "rows": res}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
