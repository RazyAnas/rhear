#!/usr/bin/env python3
"""Is the plan's evidence strong enough to bet four days of someone else's work?

The whole recommendation rests on one claim measured over 54 clips:

    no magnitude-only design reaches dSI-SDR 15 dB at 0 dB -- a PERFECT
    257-bin mask reaches only 14.05

14.05 against a target of 15.00 is a margin of 0.95 dB on 54 clips. If the
sampling error on that mean is of the same order, the claim is not safe and the
plan built on it is not either. This bootstraps the per-clip values to find out,
and does the same for the chosen plan's own margins.

Rows kept deliberately few so this runs in a couple of minutes:

    A      noisy
    R257   ideal mask @257 bins, noisy phase     <- the impossibility claim
    X96    ideal mask @96 bands + clean phase <4 kHz   <- the chosen plan

    /opt/anaconda3/bin/python E03/plan_confidence.py --data ../handoff/data/edef
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

from oracle_ladder import band_limit, score          # noqa: E402
from gtcrn_lite import N_FFT, FS                     # noqa: E402
from train_interim import Pairs, stft, istft, DEV    # noqa: E402
from rhear_data.manifest import read_manifest        # noqa: E402

TARGETS = {"stoi": 0.85, "pesq": 2.5, "dsisdr": 15.0}
BOOT = 10000
RNG = np.random.default_rng(7)


def boot_ci(vals, base, fn=np.nanmean, lo=2.5, hi=97.5):
    """Percentile bootstrap over CLIPS, resampling the row and its own baseline
    together so the dSI-SDR difference keeps its pairing."""
    v = np.asarray(vals, float)
    b = np.asarray(base, float)
    n = len(v)
    idx = RNG.integers(0, n, size=(BOOT, n))
    stat = fn(v[idx], axis=1) - fn(b[idx], axis=1)
    return float(np.percentile(stat, lo)), float(np.percentile(stat, hi))


def boot_ci_abs(vals, fn=np.nanmean, lo=2.5, hi=97.5):
    v = np.asarray(vals, float)
    n = len(v)
    idx = RNG.integers(0, n, size=(BOOT, n))
    stat = fn(v[idx], axis=1)
    return float(np.percentile(stat, lo)), float(np.percentile(stat, hi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--window", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "plan_confidence.json"))
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    test_rows = [r for r in rows if r.get("split") == "test"]
    near0 = [i for i, r in enumerate(test_rows) if abs(float(r["snr_db"])) <= a.window]
    if len(near0) > a.n:
        sel = np.linspace(0, len(near0) - 1, a.n).astype(int)
        near0 = [near0[i] for i in sel]

    NAMES = ["A noisy", "R257 mask@257, noisy phase", "X96 mask@96 + phase <4 kHz"]
    per = {k: {m: [] for m in ("stoi", "pesq", "sdr")} for k in NAMES}

    win = torch.hann_window(N_FFT, device=DEV)
    cut4k = int(round(4000 / (FS / N_FFT)))

    print(f"\ndata      {os.path.abspath(a.data)}")
    print(f"clips     {len(near0)} at 0 dB +/-{a.window} dB")
    print(f"bootstrap {BOOT:,} resamples over clips, seed 7\n")

    for done, i in enumerate(near0):
        x, y = ds[i]
        n = x.shape[-1]
        Xt = stft(x[None].to(DEV), win)[0]
        St = stft(y[None].to(DEV), win)[0]
        clean, noisy = y.numpy(), x.numpy()

        mag_x, mag_s = Xt.abs(), St.abs()
        ph_x, ph_s = Xt.angle(), St.angle()
        irm = torch.clamp(mag_s / (mag_x + 1e-8), max=1.0)

        def synth(m, ph):
            return istft((m * torch.exp(1j * ph))[None], win, n)[0].cpu().numpy()

        m96 = band_limit(irm[None], 96)[0]
        ph_mix = ph_x.clone()
        ph_mix[:cut4k] = ph_s[:cut4k]

        out = {
            "A noisy": noisy,
            "R257 mask@257, noisy phase": synth(irm * mag_x, ph_x),
            "X96 mask@96 + phase <4 kHz": synth(m96 * mag_x, ph_mix),
        }
        for k, est in out.items():
            o = score(clean, noisy, est)
            for m in per[k]:
                per[k][m].append(o[m])
        print(f"\r  clip {done + 1}/{len(near0)}", end="", flush=True)
    print("\n")

    base_sdr = per["A noisy"]["sdr"]
    report = {}

    for k in NAMES[1:]:
        print(f"{k}")
        r = {}
        for m, label in (("stoi", "STOI"), ("pesq", "PESQ")):
            mean = float(np.nanmean(per[k][m]))
            lo, hi = boot_ci_abs(per[k][m])
            t = TARGETS[m]
            safe = "clears" if lo >= t else ("STRADDLES" if hi >= t else "fails")
            print(f"  {label:8s} {mean:7.3f}   95% CI [{lo:7.3f}, {hi:7.3f}]   "
                  f"target {t:5.2f}   {safe}")
            r[m] = {"mean": mean, "ci": [lo, hi], "target": t, "verdict": safe}

        mean = float(np.nanmean(per[k]["sdr"]) - np.nanmean(base_sdr))
        lo, hi = boot_ci(per[k]["sdr"], base_sdr)
        t = TARGETS["dsisdr"]
        safe = "clears" if lo >= t else ("STRADDLES" if hi >= t else "fails")
        print(f"  {'dSI-SDR':8s} {mean:7.3f}   95% CI [{lo:7.3f}, {hi:7.3f}]   "
              f"target {t:5.2f}   {safe}")
        r["dsisdr"] = {"mean": mean, "ci": [lo, hi], "target": t, "verdict": safe}
        print()
        report[k] = r

    print("How to read this:")
    print("  'fails'      the whole interval is below target -- the claim is safe")
    print("  'STRADDLES'  the interval crosses the target -- 54 clips cannot")
    print("               settle it, and any plan resting on it is unproven")
    print("  'clears'     the whole interval is above target")

    with open(a.out, "w") as f:
        json.dump({"data": os.path.abspath(a.data), "n_clips": len(near0),
                   "bootstrap": BOOT, "targets": TARGETS, "rows": report,
                   "per_clip": {k: {m: list(map(float, v)) for m, v in d.items()}
                                for k, d in per.items()}}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
