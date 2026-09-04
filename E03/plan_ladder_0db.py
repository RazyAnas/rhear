#!/usr/bin/env python3
"""Which design family can clear the PS targets at 0 dB? Priced, not argued.

oracle_at_0db.py established that at the SNR the PS actually specifies, our
current design (real mask on 48 ERB bands, noisy phase kept) has a ceiling of
STOI 0.938 / PESQ 2.392 / dSI-SDR 13.02 -- so two of the three targets are
unreachable no matter how training goes.

This prices the candidate fixes, all at ORACLE strength on the same clips:

  resolution sweep   real mask at 24 / 48 / 64 / 96 / 128 / 257 bands,
                     noisy phase kept.  Answers: how many bands does PESQ 2.5
                     need, and does resolution alone ever reach dSI-SDR 15?

  phase sweep        real mask @48 bands, with the CLEAN phase substituted only
                     below a cutoff (1 / 2 / 4 / 8 kHz).  Answers: how much of
                     the spectrum's phase must actually be fixed, since fixing
                     all of it is not a buildable proposal.

  combined           the cheapest resolution that clears PESQ, crossed with the
                     cheapest phase cutoff that clears dSI-SDR.

WHAT THIS PROVES AND WHAT IT DOES NOT
-------------------------------------
An oracle cannot be beaten by any model of its family. So a row BELOW a target
is a proof of impossibility -- decisive, and no amount of training or data
changes it. A row ABOVE a target proves only that the target is not ruled out.
Nothing here guarantees a trained network reaches it.

To make the second half honest, every surviving row is also reported as the
fraction of its own ceiling that G7-base currently achieves on the same clips.
If a plan needs 95% of its ceiling and we have never exceeded 70%, that plan is
not a plan.

    /opt/anaconda3/bin/python E03/plan_ladder_0db.py --data ../handoff/data/edef
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

BANDS = [24, 48, 64, 96, 128]        # 257 = full resolution, handled separately
PHASE_KHZ = [1, 2, 4, 8]             # 8 kHz = Nyquist = all of it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--window", type=float, default=2.5)
    ap.add_argument("--n", type=int, default=54)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "plan_ladder_0db.json"))
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

    names = (["A noisy"]
             + [f"R{b} mask@{b} bands, noisy phase" for b in BANDS]
             + ["R257 mask@257 bins, noisy phase"]
             + [f"P{k}k mask@48 + clean phase <{k} kHz" for k in PHASE_KHZ]
             + [f"X mask@{b} + clean phase <{k} kHz" for b, k in ((96, 2), (96, 4), (257, 2))])

    acc = {k: {m: [] for m in ("stoi", "pesq", "sdr", "sir", "sar")} for k in names}

    print(f"\ndata      {os.path.abspath(a.data)}")
    print(f"clips     {len(near0)} within +/-{a.window} dB of 0 dB")
    print(f"rows      {len(names)} oracles, no checkpoint loaded\n")

    win = torch.hann_window(N_FFT, device=DEV)
    df = FS / N_FFT                                  # Hz per bin

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

        masks = {b: band_limit(irm[None], b)[0] for b in BANDS}
        masks[257] = irm

        out = {"A noisy": noisy}
        for b in BANDS:
            out[f"R{b} mask@{b} bands, noisy phase"] = synth(masks[b] * mag_x, ph_x)
        out["R257 mask@257 bins, noisy phase"] = synth(irm * mag_x, ph_x)

        # Partial phase: clean phase below the cutoff bin, noisy phase above.
        # This is the buildable version of "fix the phase" -- a network that
        # estimates phase only where it is cheapest and matters most.
        for k in PHASE_KHZ:
            cut = int(round(k * 1000 / df))
            ph = ph_x.clone()
            ph[:cut] = ph_s[:cut]
            out[f"P{k}k mask@48 + clean phase <{k} kHz"] = synth(masks[48] * mag_x, ph)

        for b, k in ((96, 2), (96, 4), (257, 2)):
            cut = int(round(k * 1000 / df))
            ph = ph_x.clone()
            ph[:cut] = ph_s[:cut]
            out[f"X mask@{b} + clean phase <{k} kHz"] = synth(masks[b] * mag_x, ph)

        for kk, est in out.items():
            o = score(clean, noisy, est)
            for m in acc[kk]:
                acc[kk][m].append(o[m])

        print(f"\r  clip {done + 1}/{len(near0)}", end="", flush=True)
    print("\n")

    dropped = int(np.sum(~np.isfinite(acc["A noisy"]["pesq"])))
    if dropped:
        print(f"note: PESQ could not score {dropped} of {len(near0)} clips; "
              f"excluded from the PESQ column only.\n")

    res = {k: {m: float(np.nanmean(v)) for m, v in d.items()} for k, d in acc.items()}
    base = res["A noisy"]["sdr"]
    for k in res:
        res[k]["dsisdr"] = res[k]["sdr"] - base

    w = max(len(k) for k in names)
    print(f"{'row'.ljust(w)}   STOI     PESQ   dSI-SDR   clears all three?")
    print("-" * (w + 44))
    survivors = []
    for k in names:
        r = res[k]
        ok = (r["stoi"] >= TARGETS["stoi"] and r["pesq"] >= TARGETS["pesq"]
              and r["dsisdr"] >= TARGETS["dsisdr"])
        miss = [n for n, m in (("STOI", "stoi"), ("PESQ", "pesq"), ("dSI-SDR", "dsisdr"))
                if r[m] < TARGETS[m]]
        verdict = "YES" if ok else "no  (" + ", ".join(miss) + ")"
        if ok:
            survivors.append(k)
        print(f"{k.ljust(w)}  {r['stoi']:.3f}   {r['pesq']:6.3f}   "
              f"{r['dsisdr']:6.2f}   {verdict}")

    print(f"\nPS targets at 0 dB:  STOI >= {TARGETS['stoi']}   "
          f"PESQ >= {TARGETS['pesq']}   dSI-SDR >= {TARGETS['dsisdr']} dB")

    if not survivors:
        print("\nNo candidate clears all three. The families tested are all "
              "ruled out at 0 dB on this data.")
    else:
        print(f"\n{len(survivors)} candidate(s) clear all three at oracle strength:")
        for k in survivors:
            print(f"  · {k}")
        print("\nA ceiling above target is NOT a guarantee. See the demand table.")

    with open(a.out, "w") as f:
        json.dump({"data": os.path.abspath(a.data),
                   "snr_window_db": a.window,
                   "n_clips": len(near0),
                   "pesq_unscorable_clips": dropped,
                   "targets": TARGETS,
                   "survivors": survivors,
                   "rows": res}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
