#!/usr/bin/env python3
"""The oracle ladder, restricted to the condition the PS actually specifies.

Every acceptance target in docs/02-architecture.md section 10 is written
"at 0 dB input SNR":

    STOI >= 0.85      PESQ >= 2.5      delta SI-SDR >= 15 dB

Every oracle number we have ever computed -- and every model number we have
ever reported -- is an average over a -10..+20 dB mix. Those two things are not
comparable, and the direction of the error is not obvious in advance: the mix
average is flattered by the easy high-SNR clips on PESQ and STOI, but delta
SI-SDR is *larger* at low SNR, so the mix understates it. Averaging across the
mix and comparing to a target defined at a point is a category error either way.

This runs rows A / B / F / C / D of oracle_ladder.py on the clips whose manifest
snr_db is within +/- 2.5 dB of zero, so the ceiling is measured at the condition
we are actually graded on. It trains nothing and loads no checkpoint -- these
are oracles, so no model of that family can beat them.

    /opt/anaconda3/bin/python E03/oracle_at_0db.py --data ../handoff/data/edef

Reading the result: any target that sits ABOVE row B is unreachable for our
current design (a real-valued mask on 48 ERB bands with the noisy phase kept),
no matter how the training goes. Row F prices more frequency resolution. Rows C
and D price the phase, which is the only thing that separates F from D.
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

# score() is reused verbatim from the ladder so these numbers are computed by
# exactly the same code path as every oracle number already in the repo.
from oracle_ladder import band_limit, score              # noqa: E402
from gtcrn_lite import N_FFT, FS                         # noqa: E402
from train_interim import Pairs, stft, istft, DEV        # noqa: E402
from rhear_data.manifest import read_manifest            # noqa: E402

ROWS = [
    "A noisy",
    "B ideal@48, noisy phase",
    "F ideal@257, noisy phase",
    "C ideal@48, CLEAN phase",
    "D ideal@257, CLEAN phase",
]

TARGETS = {"stoi": 0.85, "pesq": 2.5, "dsisdr": 15.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--window", type=float, default=2.5,
                    help="keep clips with |snr_db| <= this. 2.5 dB gives a band "
                         "around 0 dB wide enough to hold a usable sample.")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)

    # Pairs keeps the test rows in order, so the manifest's test rows line up
    # with dataset indices one for one.
    test_rows = [r for r in rows if r.get("split") == "test"]
    if len(test_rows) != len(ds):
        sys.exit(
            f"manifest has {len(test_rows)} test rows but the dataset reports "
            f"{len(ds)}. Refusing to index one by the other."
        )

    near0 = [i for i, r in enumerate(test_rows)
             if abs(float(r["snr_db"])) <= a.window]
    if not near0:
        sys.exit(f"No test clips within +/-{a.window} dB of 0 dB.")

    if len(near0) > a.n:
        sel = np.linspace(0, len(near0) - 1, a.n).astype(int)
        near0 = [near0[i] for i in sel]

    snrs = [float(test_rows[i]["snr_db"]) for i in near0]
    print(f"\ndata      {os.path.abspath(a.data)}")
    print(f"clips     {len(near0)} within +/-{a.window} dB of 0 dB "
          f"(actual {min(snrs):+.2f} .. {max(snrs):+.2f}, mean {np.mean(snrs):+.2f})")
    print(f"rows      {len(ROWS)} oracles, no checkpoint loaded\n")

    win = torch.hann_window(N_FFT, device=DEV)
    acc = {k: {m: [] for m in ("stoi", "pesq", "sdr", "sir", "sar")} for k in ROWS}

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

        m48 = band_limit(irm[None], 48)[0]
        out = {
            "A noisy": noisy,
            "B ideal@48, noisy phase": synth(m48 * mag_x, ph_x),
            "F ideal@257, noisy phase": synth(irm * mag_x, ph_x),
            "C ideal@48, CLEAN phase": synth(m48 * mag_x, ph_s),
            "D ideal@257, CLEAN phase": synth(irm * mag_x, ph_s),
        }

        for k, est in out.items():
            o = score(clean, noisy, est)
            for m in acc[k]:
                acc[k][m].append(o[m])

        print(f"\r  {done + 1}/{len(near0)}", end="", flush=True)
    print("\n")

    # score() returns nan for a clip PESQ refuses (typically too little speech
    # to score). Averaging with np.mean would turn one such clip into a nan
    # column and silently hide the whole result, so average what scored and say
    # how many did not.
    dropped = {k: int(np.sum(~np.isfinite(d["pesq"]))) for k, d in acc.items()}
    if dropped["A noisy"]:
        print(f"note: PESQ could not score {dropped['A noisy']} of {len(near0)} "
              f"clips; those are excluded from the PESQ column only.\n")

    res = {k: {m: float(np.nanmean(v)) for m, v in d.items()} for k, d in acc.items()}
    base_sdr = res["A noisy"]["sdr"]
    for k in res:
        res[k]["dsisdr"] = res[k]["sdr"] - base_sdr

    w = max(len(k) for k in ROWS)
    print(f"{'row'.ljust(w)}   STOI     PESQ    SI-SDR   dSI-SDR    SIR     SAR")
    print("-" * (w + 56))
    for k in ROWS:
        r = res[k]
        print(f"{k.ljust(w)}  {r['stoi']:.3f}   {r['pesq']:6.3f}   "
              f"{r['sdr']:6.2f}   {r['dsisdr']:6.2f}   {r['sir']:6.2f}  {r['sar']:6.2f}")

    print(f"\nPS targets at 0 dB:  STOI >= {TARGETS['stoi']}   "
          f"PESQ >= {TARGETS['pesq']}   dSI-SDR >= {TARGETS['dsisdr']} dB\n")

    b = res["B ideal@48, noisy phase"]
    print("Against row B -- the ceiling of the design we are actually building:")
    for m, label in (("stoi", "STOI"), ("pesq", "PESQ"), ("dsisdr", "dSI-SDR")):
        head = b[m] - TARGETS[m]
        verdict = "reachable" if head >= 0 else "UNREACHABLE by this design"
        print(f"  {label:8s} ceiling {b[m]:7.3f}   target {TARGETS[m]:6.2f}   "
              f"headroom {head:+7.3f}   {verdict}")

    print("\nWhat the other rows buy, on top of row B:")
    for k in ("F ideal@257, noisy phase", "C ideal@48, CLEAN phase",
              "D ideal@257, CLEAN phase"):
        r = res[k]
        print(f"  {k:28s}  PESQ {r['pesq'] - b['pesq']:+.3f}   "
              f"STOI {r['stoi'] - b['stoi']:+.3f}   dSI-SDR {r['dsisdr'] - b['dsisdr']:+.2f}")

    out_path = a.out or os.path.join(HERE, "runs", "g012", "oracle_at_0db.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"data": os.path.abspath(a.data),
                   "snr_window_db": a.window,
                   "n_clips": len(near0),
                   "snr_actual": {"min": min(snrs), "max": max(snrs),
                                  "mean": float(np.mean(snrs))},
                   "targets": TARGETS,
                   "rows": res}, f, indent=1)
    print(f"\nwritten   {out_path}")


if __name__ == "__main__":
    main()
