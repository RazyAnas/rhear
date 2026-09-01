#!/usr/bin/env python3
"""Isolate what the learned phase branch contributes.

The model emits, per gtcrn_lite.forward + train_interim.enhance:

    S_hat(t,f) = tanh(|m|) * |X| * exp( j*( angle X + angle m ) )

`angle m` is produced at 48 ERB bands and then expanded to 257 bins, so ONE
rotation is applied to every bin in a band. Phase is circular and wraps
quickly across frequency, and our measured band widths reach 19 bins, so a
band-constant rotation is ill-posed in the wide bands.

This holds the model's own MAGNITUDE mask fixed and varies only the phase:

    predicted   angle X + angle m      what ships today
    zero        angle X                the phase branch switched off
    oracle      angle S                the clean phase, an upper bound

If `zero` beats `predicted`, the branch is actively harmful and removing it
is a free improvement that also drops Atan -- one of the operators CMSIS-NN
does not support.
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

VARIANTS = ("predicted (ships today)", "zero (branch off)", "oracle (clean)")


def run(data, ckpt, n):
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, "test", rows, cache=False)
    meta = [r for r in rows if r["split"] == "test"]
    m = GTCRNLite().to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV))
    m.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    sel = range(len(ds)) if n is None else \
        np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)

    out = []
    for i in sel:
        x, y = ds[int(i)]
        c = y.numpy().astype(np.float64)
        nz = x.numpy().astype(np.float64)
        noise = nz - c
        X, S = stft(x[None].to(DEV), win), stft(y[None].to(DEV), win)
        spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
        with torch.no_grad():
            mm, mp, _ = m(spec)
        mag = (mm * X.abs().transpose(1, 2).unsqueeze(1)).squeeze(1).transpose(1, 2)
        phases = {
            VARIANTS[0]: X.angle() + mp.squeeze(1).transpose(1, 2),
            VARIANTS[1]: X.angle(),
            VARIANTS[2]: S.angle(),
        }
        r = dict(meta[int(i)])
        r["mp_abs"] = float(mp.abs().mean())
        for k, ph in phases.items():
            e = istft(mag * torch.exp(1j * ph), win,
                      x.shape[-1])[0].cpu().numpy().astype(np.float64)
            L = min(len(e), len(c))
            st, pq = score(c[:L], e[:L])
            sd, si, sa = si_metrics(e[:L], c[:L], noise[:L])
            r.update({f"{k}|stoi": st, f"{k}|pesq": pq, f"{k}|sdr": sd,
                      f"{k}|sir": si, f"{k}|sar": sa})
        out.append(r)
    return out


def agg(rows, k):
    f = lambda mname: float(np.nanmean([r[f"{k}|{mname}"] for r in rows]))
    return {mm: f(mm) for mm in ("stoi", "pesq", "sdr", "sir", "sar")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/h3_20k/best.pt")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="runs/h3_20k/phase_isolation.json")
    a = ap.parse_args()

    res = {}
    for d in a.sets:
        tag = os.path.basename(d.rstrip("/"))
        rows = run(d, a.ckpt, a.n)
        print("\n=== %s   (%d clips, ckpt %s) ===" % (tag, len(rows), a.ckpt))
        print("  %-26s%8s%8s%9s%9s%9s"
              % ("phase source", "STOI", "PESQ", "SI-SDR", "SI-SIR", "SI-SAR"))
        print("  " + "-" * 69)
        got = {}
        for k in VARIANTS:
            v = agg(rows, k)
            got[k] = v
            print("  %-26s%8.3f%8.3f%9.2f%9.2f%9.2f"
                  % (k, v["stoi"], v["pesq"], v["sdr"], v["sir"], v["sar"]))
        p, z = got[VARIANTS[0]], got[VARIANTS[1]]
        print("  %-26s%+8.3f%+8.3f%+9.2f%+9.2f%+9.2f"
              % ("zero - predicted", z["stoi"] - p["stoi"], z["pesq"] - p["pesq"],
                 z["sdr"] - p["sdr"], z["sir"] - p["sir"], z["sar"] - p["sar"]))
        print("  mean |predicted rotation| = %.4f rad (%.1f deg)"
              % (np.mean([r["mp_abs"] for r in rows]),
                 np.degrees(np.mean([r["mp_abs"] for r in rows]))))
        win_all = (z["stoi"] > p["stoi"] and z["pesq"] > p["pesq"]
                   and z["sar"] > p["sar"])
        print("  -> %s" % ("branch is HARMFUL on all three (STOI, PESQ, SI-SAR)"
                           if win_all else
                           "branch is NOT uniformly harmful -- hypothesis not confirmed"))
        res[tag] = {"variants": got, "confirmed": bool(win_all)}

    json.dump(res, open(a.out, "w"), indent=1)
    print("\n  -> %s" % a.out)
    print("  VERDICT: %s" % ("CONFIRMED on every set -- proceed to retrain "
                             "without the phase branch"
                             if all(v["confirmed"] for v in res.values())
                             else "NOT confirmed on every set -- reassess"))
