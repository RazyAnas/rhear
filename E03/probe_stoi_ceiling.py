#!/usr/bin/env python3
"""What is the BEST STOI this architecture could ever reach?

Applies an ORACLE mask -- computed from the true clean signal, which no trained
model can beat -- at two resolutions:

  full  : one mask value per STFT bin (257)          the upper bound for any mask
  ERB   : one mask value per ERB band (48), expanded  the bound OUR model has

If the ERB-limited oracle barely beats the noisy signal, the mask resolution is
the ceiling and no amount of training moves STOI. If it is far above, the model
is simply undertrained or too small. This is a ceiling measurement, not a guess.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import ERBSplit, N_FFT, HOP, FS
from train_interim import Pairs
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--n", type=int, default=80)
ap.add_argument("--bands", type=int, nargs="+", default=[24, 48, 96, 257])
a = ap.parse_args()

hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
ds = Pairs(a.data, "test", rows, cache=False)
win = torch.hann_window(N_FFT)
idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)

def run(nb):
    erb = ERBSplit(n_bands=nb) if nb < 257 else None
    S, P = [], []
    for i in idx:
        x, y = ds[int(i)]
        X = torch.stft(x[None], N_FFT, HOP, N_FFT, win, return_complex=True, center=True)
        Y = torch.stft(y[None], N_FFT, HOP, N_FFT, win, return_complex=True, center=True)
        m = (Y.abs() / (X.abs() + 1e-8)).clamp(0, 1)          # oracle ratio mask
        if erb is not None:                                    # limit resolution
            mm = m.transpose(1, 2).unsqueeze(1)                # (1,1,T,F)
            mm = erb.inverse(erb(mm))
            mm = mm / (erb.inverse(erb(torch.ones_like(mm))) + 1e-8)
            m = mm.squeeze(1).transpose(1, 2)
        est = torch.istft(X * m, N_FFT, HOP, N_FFT, win, center=True, length=x.shape[-1])
        c = y.numpy().astype(np.float64); e = est[0].numpy().astype(np.float64)
        try: S.append(stoi_fn(c, e, FS, extended=False))
        except Exception: pass
        try: P.append(pesq_fn(FS, c, e, "wb"))
        except Exception: pass
    return float(np.mean(S)), float(np.mean(P))

# reference: the noisy signal itself
Sn, Pn = [], []
for i in idx:
    x, y = ds[int(i)]
    c = y.numpy().astype(np.float64); n = x.numpy().astype(np.float64)
    try: Sn.append(stoi_fn(c, n, FS, extended=False))
    except Exception: pass
    try: Pn.append(pesq_fn(FS, c, n, "wb"))
    except Exception: pass
print(f"  {len(idx)} test clips\n")
print(f"  {'mask resolution':<24}{'STOI':>8}{'PESQ':>8}{'d STOI':>9}")
print("  " + "-" * 49)
print(f"  {'noisy (no processing)':<24}{np.mean(Sn):8.3f}{np.mean(Pn):8.3f}{0.0:9.3f}")
out = {"noisy": {"stoi": float(np.mean(Sn)), "pesq": float(np.mean(Pn))}}
for nb in a.bands:
    s, p = run(nb)
    lbl = f"ORACLE mask @ {nb} bands" if nb < 257 else "ORACLE mask @ full 257 bins"
    print(f"  {lbl:<24}{s:8.3f}{p:8.3f}{s-np.mean(Sn):+9.3f}")
    out[f"oracle_{nb}"] = {"stoi": s, "pesq": p}
json.dump(out, open("runs/interim/stoi_ceiling.json", "w"), indent=1)
print()
o48 = out["oracle_48"]["stoi"]; o257 = out["oracle_257"]["stoi"]
print(f"  ceiling for OUR architecture (48 ERB bands): STOI {o48:.3f}")
print(f"  ceiling for any full-resolution mask:        STOI {o257:.3f}")
print(f"  resolution costs {o257-o48:+.3f} STOI")
print(f"  PS target is 0.85 -> {'REACHABLE' if o48>=0.85 else 'NOT REACHABLE'} at 48 bands")
