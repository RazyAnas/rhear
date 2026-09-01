#!/usr/bin/env python3
"""PHASE 4 - what ceiling can each mask family reach, per unit of compute?

The measured 2.96 PESQ ceiling at 48 bands is the ceiling of a REAL-VALUED
GAIN per band. It is not the ceiling of the band count. Two ways to raise it:

  more bands      48 -> 96: doubles the ERB matrices, the mask head, and the
                  final decoder stage.
  deep filtering  keep 48 bands, but predict a short COMPLEX FIR per band,
                  applied across time. Can rotate phase and resolve
                  sub-band structure that one real gain cannot.

Oracle formulation for deep filtering. For band b and frame t, solve for ONE
complex FIR w (K taps) shared by every bin in that band -- which is what a
model would actually predict:

    minimise  sum_{f in band b}  | Y[f,t] - sum_k w_k X[f, t-k] |^2

by ridge-regularised least squares. K=1 is a complex gain per band, already
stronger than a real gain because it can correct phase.

CAVEAT, stated because it bounds how much this number is worth: low-frequency
ERB bands hold only 1-2 bins, so there the system is exactly solvable and the
oracle collapses to full resolution. The real-gain oracle has the same
property, so the comparison stays fair, but neither is reachable by a trained
model. Bins-per-band is reported so the reader can judge.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import ERBSplit, N_FFT, HOP, FS
from train_interim import Pairs
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

WIN = torch.hann_window(N_FFT)


def band_bins(nb):
    """Bin indices belonging to each ERB band."""
    M = ERBSplit(n_bands=nb).M if hasattr(ERBSplit(n_bands=nb), "M") else None
    erb = ERBSplit(n_bands=nb)
    M = dict(erb.named_buffers()).get("M", None)
    if M is None:
        for _, v in erb.named_buffers():
            if v.ndim == 2:
                M = v
                break
    assert M is not None, "could not recover the ERB matrix"
    return [torch.nonzero(M[b] > 0).flatten() for b in range(nb)]


def stft(x):
    return torch.stft(x[None], N_FFT, HOP, N_FFT, WIN,
                      return_complex=True, center=True)


def istft(X, n):
    return torch.istft(X, N_FFT, HOP, N_FFT, WIN, center=True, length=n)


def oracle_real_band(X, Y, nb):
    """Current family: one real gain per band, clamped to [0,1]."""
    erb = ERBSplit(n_bands=nb)
    m = (Y.abs() / (X.abs() + 1e-8)).clamp(0, 1)
    mm = m.transpose(1, 2).unsqueeze(1)
    mm = erb.inverse(erb(mm))
    mm = mm / (erb.inverse(erb(torch.ones_like(mm))) + 1e-8)
    return X * mm.squeeze(1).transpose(1, 2)


def oracle_deep_filter(X, Y, nb, K, ridge=1e-6):
    """One complex FIR of K taps per band, shared across the band's bins."""
    bins = band_bins(nb)
    F, T = X.shape[1], X.shape[2]
    Xp = torch.cat([torch.zeros(1, F, K - 1, dtype=X.dtype), X], dim=2)
    out = torch.zeros_like(X)
    for b, idx in enumerate(bins):
        if idx.numel() == 0:
            continue
        # A[t, i, k] = X[idx_i, t - k]
        A = torch.stack([Xp[0, idx, (K - 1) - k: (K - 1) - k + T]
                         for k in range(K)], dim=-1)      # (n_b, T, K)
        A = A.permute(1, 0, 2)                            # (T, n_b, K)
        y = Y[0, idx, :].permute(1, 0).unsqueeze(-1)      # (T, n_b, 1)
        Ah = A.conj().transpose(-2, -1)                   # (T, K, n_b)
        G = Ah @ A                                        # (T, K, K)
        sc = G.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1e-12)
        G = G + (ridge * sc)[:, None, None] * torch.eye(K, dtype=G.dtype)
        w = torch.linalg.solve(G, Ah @ y)                 # (T, K, 1)
        out[0, idx, :] = (A @ w).squeeze(-1).permute(1, 0)
    return out


def score(clean, est):
    c = clean.astype(np.float64)
    e = est.astype(np.float64)
    s = p = np.nan
    try:
        s = stoi_fn(c, e, FS, extended=False)
    except Exception:
        pass
    try:
        p = pesq_fn(FS, c, e, "wb")
    except Exception:
        pass
    return s, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out", default="runs/interim/phase4_ceiling.json")
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
    print("  %d clips from %s\n" % (len(idx), a.data))

    for nb in (48, 96):
        bb = [len(x) for x in band_bins(nb)]
        print("  %3d bands: bins/band min %d, median %d, max %d  "
              "(%d of %d bands hold < 3 bins)"
              % (nb, min(bb), int(np.median(bb)), max(bb),
                 sum(1 for v in bb if v < 3), nb))
    print()

    # mask values a model must emit per frame -- the compute proxy
    cost = {"noisy": 0, "real_48": 48, "real_96": 96, "real_257": 257,
            "df48_K1": 48 * 2, "df48_K2": 48 * 4, "df48_K3": 48 * 6}
    fams = [("noisy", None), ("real_48", ("real", 48)),
            ("real_96", ("real", 96)), ("real_257", ("real", 257)),
            ("df48_K1", ("df", 48, 1)), ("df48_K2", ("df", 48, 2)),
            ("df48_K3", ("df", 48, 3))]

    acc = {k: ([], []) for k, _ in fams}
    for j, i in enumerate(idx):
        x, y = ds[int(i)]
        X, Y = stft(x), stft(y)
        for name, spec in fams:
            if spec is None:
                est = x.numpy()
            elif spec[0] == "real" and spec[1] == 257:
                m = (Y.abs() / (X.abs() + 1e-8)).clamp(0, 1)
                est = istft(X * m, x.shape[-1])[0].numpy()
            elif spec[0] == "real":
                est = istft(oracle_real_band(X, Y, spec[1]),
                            x.shape[-1])[0].numpy()
            else:
                est = istft(oracle_deep_filter(X, Y, spec[1], spec[2]),
                            x.shape[-1])[0].numpy()
            s, p = score(y.numpy(), est)
            acc[name][0].append(s)
            acc[name][1].append(p)
        if (j + 1) % 20 == 0:
            print("    %d/%d" % (j + 1, len(idx)), flush=True)

    base_s = float(np.nanmean(acc["noisy"][0]))
    out = {}
    print("\n  %-14s%8s%8s%10s%10s%12s" %
          ("family", "STOI", "PESQ", "dSTOI", "dPESQ", "vals/frame"))
    print("  " + "-" * 62)
    for name, _ in fams:
        s = float(np.nanmean(acc[name][0]))
        p = float(np.nanmean(acc[name][1]))
        out[name] = {"stoi": s, "pesq": p, "vals_per_frame": cost[name]}
        print("  %-14s%8.3f%8.3f%+10.3f%+10.3f%12d"
              % (name, s, p, s - base_s,
                 p - float(np.nanmean(acc["noisy"][1])), cost[name]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("\n  -> %s" % a.out)
    print("\n  PS targets: STOI 0.85, PESQ 2.5")
    for name in ("real_48", "real_96", "df48_K2", "df48_K3"):
        o = out[name]
        print("    %-10s PESQ ceiling %.2f  %s"
              % (name, o["pesq"],
                 "clears 2.5" if o["pesq"] >= 2.5 else "BELOW 2.5"))
