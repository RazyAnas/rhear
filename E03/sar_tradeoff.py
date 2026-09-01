#!/usr/bin/env python3
"""Trade noise suppression for speech quality, and find where the optimum is.

The goal: keep ~20 dB of suppression while recovering the missing speech
quality. We can afford to. Measured on the frozen checkpoint, E_def:

    SI-SIR  20.2 dB   noise removed        -- requirement is 15 dB, so 5 dB spare
    SI-SAR  10.5 dB   damage caused        -- ceiling on this grid is 22.9 dB
    SI-SDR   9.6 dB   net, capped by SAR

and the mask is very aggressive: median gain 0.090, 51.6% of bins attenuated
by more than 20 dB. Over-suppression is the obvious suspect for the artefacts.

So: soften the mask at INFERENCE on the frozen weights and sweep it. Two knobs,
both standard, both bounded so the output can still only attenuate:

    floor  m' = max(m, f)        stop any bin being driven to zero
    power  m' = m ** g,  g < 1   lift small values, leave m=1 untouched

No retraining. If a setting raises PESQ and SI-SAR while holding SI-SIR at or
above 15 dB, that is the answer to the goal and it costs nothing to deploy.

NOTE ON A PRIOR RESULT. We tested gain floors before and recorded that they
hurt. That was judged on SI-SDR and STOI, before we had the SIR/SAR split, and
SI-SDR mixes the two effects this experiment is trying to separate. Re-running
it against SAR is not ignoring the earlier result -- it is asking the question
the earlier metric could not answer.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import GTCRNLite, N_FFT, FS
from train_interim import Pairs, stft, istft, DEV
from rhear_data.manifest import read_manifest
from phase5_comb import si_metrics, score

SIR_FLOOR_DB = 15.0          # the PS requirement we must not fall below


def run(data, ckpt, n, settings):
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, "test", rows, cache=False)
    m = GTCRNLite().to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV))
    m.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    acc = {k: {q: [] for q in ("stoi", "pesq", "sdr", "sir", "sar")}
           for k in settings}

    for i in idx:
        x, y = ds[int(i)]
        c = y.numpy().astype(np.float64)
        nz = x.numpy().astype(np.float64)
        noise = nz - c
        X = stft(x[None].to(DEV), win)
        spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
        with torch.no_grad():
            mm, mp, _ = m(spec)
        ph = X.angle()
        if mp is not None:
            ph = ph + mp.squeeze(1).transpose(1, 2)
        base = mm.squeeze(1).transpose(1, 2)          # (B,F,T) in [0,1)
        mag = X.abs()
        for key, (f, g, sm) in settings.items():
            mk = base ** g if g != 1.0 else base
            if f > 0:
                mk = torch.clamp(mk, min=f)
            if sm > 0:
                # One-pole smoothing ACROSS FRAMES. Musical noise comes from
                # the gain of a band jumping between frames; averaging the
                # mask in time is the classic remedy and is a different
                # mechanism from reshaping the gain values.
                out = torch.empty_like(mk)
                prev = mk[:, :, 0]
                out[:, :, 0] = prev
                for t in range(1, mk.shape[2]):
                    prev = sm * prev + (1 - sm) * mk[:, :, t]
                    out[:, :, t] = prev
                mk = out
            e = istft(mk * mag * torch.exp(1j * ph), win,
                      x.shape[-1])[0].cpu().numpy().astype(np.float64)
            L = min(len(e), len(c))
            st, pq = score(c[:L], e[:L])
            sd, si, sa = si_metrics(e[:L], c[:L], noise[:L])
            for nm, v in zip(("stoi", "pesq", "sdr", "sir", "sar"),
                             (st, pq, sd, si, sa)):
                acc[key][nm].append(v)
    return {k: {q: float(np.nanmean(v)) for q, v in d.items()}
            for k, d in acc.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="runs/h9_frozen/h3_snapshot.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="runs/h9_frozen/sar_tradeoff.json")
    a = ap.parse_args()

    settings = {"baseline": (0.0, 1.0, 0.0)}
    for f in (0.02, 0.05, 0.10, 0.15):
        settings["floor f=%.2f" % f] = (f, 1.0, 0.0)
    for g in (0.8, 0.6, 0.5):
        settings["power g=%.1f" % g] = (0.0, g, 0.0)
    for sm in (0.3, 0.5, 0.7, 0.85):
        settings["time-smooth a=%.2f" % sm] = (0.0, 1.0, sm)
    settings["smooth 0.5 + floor 0.02"] = (0.02, 1.0, 0.5)
    settings["smooth 0.7 + floor 0.02"] = (0.02, 1.0, 0.7)

    res = run(a.data, a.ckpt, a.n, settings)
    base = res["baseline"]

    print("  %d clips | %s\n" % (a.n, a.ckpt))
    print("  %-26s%8s%8s%9s%9s%9s%8s" %
          ("setting", "STOI", "PESQ", "SI-SDR", "SI-SIR", "SI-SAR", "SIR>=15"))
    print("  " + "-" * 78)
    for k, v in res.items():
        ok = "yes" if v["sir"] >= SIR_FLOOR_DB else "NO"
        print("  %-26s%8.3f%8.3f%9.2f%9.2f%9.2f%8s"
              % (k, v["stoi"], v["pesq"], v["sdr"], v["sir"], v["sar"], ok))

    print("\n  change vs baseline (only settings that keep SI-SIR >= %.0f dB):"
          % SIR_FLOOR_DB)
    print("  %-26s%9s%9s%9s%9s" % ("", "dSTOI", "dPESQ", "dSI-SAR", "dSI-SIR"))
    best, best_k = None, None
    for k, v in res.items():
        if k == "baseline" or v["sir"] < SIR_FLOOR_DB:
            continue
        d = (v["stoi"] - base["stoi"], v["pesq"] - base["pesq"],
             v["sar"] - base["sar"], v["sir"] - base["sir"])
        print("  %-26s%+9.3f%+9.3f%+9.2f%+9.2f" % (k, *d))
        # rank on PESQ gain, the metric we are furthest from
        if v["pesq"] > (best or -9e9):
            best, best_k = v["pesq"], k

    print()
    if best_k and best > base["pesq"]:
        v = res[best_k]
        print("  BEST WITHIN BUDGET: %s" % best_k)
        print("    PESQ  %.3f -> %.3f  (%+.3f)" % (base["pesq"], v["pesq"],
                                                   v["pesq"] - base["pesq"]))
        print("    STOI  %.3f -> %.3f  (%+.3f)" % (base["stoi"], v["stoi"],
                                                   v["stoi"] - base["stoi"]))
        print("    SI-SAR %.2f -> %.2f dB  (%+.2f)" % (base["sar"], v["sar"],
                                                       v["sar"] - base["sar"]))
        print("    SI-SIR %.2f -> %.2f dB  (%+.2f, still above %.0f)"
              % (base["sir"], v["sir"], v["sir"] - base["sir"], SIR_FLOOR_DB))
        print("    Costs nothing to deploy: it is a transform on the mask the")
        print("    network already emits, applied at inference.")
    else:
        print("  NO setting beat the baseline on PESQ while holding SI-SIR >= %.0f dB."
              % SIR_FLOOR_DB)
        print("  Over-suppression is therefore NOT the source of the artefacts,")
        print("  and the fix has to come from training, not from a mask transform.")

    json.dump({"baseline": base, "settings": res, "sir_floor_db": SIR_FLOOR_DB},
              open(a.out, "w"), indent=1)
    print("\n  -> %s" % a.out)
