#!/usr/bin/env python3
"""PART E - inference-time ablations on the FROZEN checkpoint. No retraining.

The model is run ONCE per clip; every variant is applied to the same predicted
mask, so differences are the variant and nothing else.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
sys.path.insert(0, os.path.join(HERE, "partC"))
from gtcrn_lite import GTCRNLite, N_FFT, HOP, FS
from train_interim import Pairs, stft, istft, si_sdr, DEV
from rhear_data.manifest import read_manifest
from forensics import regions, db
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn
EPS = 1e-10


def smooth_t(m, k):
    if k <= 1: return m
    ker = torch.ones(1, 1, 1, k, device=m.device) / k
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(m[None, None], (k - 1, 0, 0, 0), mode="replicate"),
        ker)[0, 0]


def variants(m, spp_f, freqs_t):
    """m: (F,T) predicted magnitude mask. Returns {name: modified mask}."""
    V = {"A0 baseline": m}
    for al in (0.7, 0.5):
        V[f"A2 softer m^{al}"] = m ** al
    for g in (0.1, 0.2, 0.3):
        V[f"A3 floor {g}"] = torch.clamp(m, min=g)
    sp_band = (freqs_t >= 300) & (freqs_t <= 3400)
    for g in (0.2, 0.3):
        mm = m.clone()
        mm[sp_band] = torch.clamp(mm[sp_band], min=g)
        V[f"A4 speech-band floor {g}"] = mm
    for k in (3, 5):
        V[f"A5 smooth {k} frames"] = smooth_t(m, k)
    for g in (0.2, 0.3):
        fl = g * spp_f[None, :]                      # floor only where speech likely
        V[f"A6 SPP floor {g}"] = torch.maximum(m, fl)
    for g in (0.25, 0.4):
        # A7: raise the floor when the frame looks low-SNR, judged by how hard
        # the model is already suppressing that frame (its own mask level).
        lvl = m.mean(0, keepdim=True)
        w = torch.clamp((0.45 - lvl) / 0.45, 0, 1)   # 0 when confident, 1 when suppressing hard
        V[f"A7 low-SNR floor {g}"] = torch.maximum(m, g * w)
    return V


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="baseline_frozen/model.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="partC/ablations.json")
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    model = GTCRNLite().to(DEV)
    model.load_state_dict(torch.load(a.ckpt, map_location=DEV)); model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)
    ft = torch.tensor(freqs, dtype=torch.float32, device=DEV)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)

    acc, mask_stats = {}, []
    for c, i in enumerate(idx):
        x, y = ds[int(i)]
        xd = x[None].to(DEV)
        X = stft(xd, win); Y = stft(y[None].to(DEV), win)
        spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
        with torch.no_grad():
            mm, mp, spp = model(spec)
        mag = X.abs().transpose(1, 2).unsqueeze(1)
        ph = X.angle().transpose(1, 2).unsqueeze(1)
        m = mm.squeeze(1).transpose(1, 2)[0]                       # (F,T)
        mask_stats.append(m.flatten().cpu().numpy())
        # spp is (1,1,T,bands) -> per-frame speech likelihood
        spp_f = spp.squeeze(1)[0].mean(-1)                          # (T,)
        Ym = Y.abs()[0].cpu().numpy(); Xm = X.abs()[0].cpu().numpy()
        sp, nz = regions(Ym)
        cnp = y.numpy().astype(np.float64); nnp = x.numpy().astype(np.float64)

        V = variants(m, spp_f, ft)
        V["A1 noisy phase"] = m                                    # phase handled below
        for name, mv in V.items():
            use_pred_phase = (name != "A1 noisy phase")
            mv4 = mv.transpose(0, 1)[None, None]      # (F,T) -> (1,1,T,F) to match mag
            Se = ((mv4 * mag) * torch.exp(1j * (ph + mp if use_pred_phase else ph)))
            Se = Se.squeeze(1).transpose(1, 2)
            e = istft(Se, win, xd.shape[-1])[0].cpu().numpy().astype(np.float64)
            Em = np.abs(Se[0].cpu().numpy())
            g = Em / (Xm + EPS)
            d = acc.setdefault(name, dict(stoi=[], pesq=[], sisdr=[], gs=[], gn=[]))
            try: d["stoi"].append(stoi_fn(cnp, e, FS))
            except Exception: pass
            try: d["pesq"].append(pesq_fn(FS, cnp, e, "wb"))
            except Exception: pass
            d["sisdr"].append(float(si_sdr(torch.from_numpy(e)[None],
                                           torch.from_numpy(cnp)[None])))
            d["gs"].append(float(np.mean(g[sp])) if sp.any() else np.nan)
            d["gn"].append(float(np.mean(g[nz])) if nz.any() else np.nan)
        if c == 0:
            base = dict(stoi=stoi_fn(cnp, nnp, FS), pesq=pesq_fn(FS, cnp, nnp, "wb"))
        nz_ref = acc.setdefault("__noisy__", dict(stoi=[], pesq=[], sisdr=[], gs=[], gn=[]))
        try: nz_ref["stoi"].append(stoi_fn(cnp, nnp, FS))
        except Exception: pass
        try: nz_ref["pesq"].append(pesq_fn(FS, cnp, nnp, "wb"))
        except Exception: pass
        nz_ref["sisdr"].append(float(si_sdr(torch.from_numpy(nnp)[None],
                                            torch.from_numpy(cnp)[None])))
        nz_ref["gs"].append(1.0); nz_ref["gn"].append(1.0)
        if c % 40 == 0: print(f"    {c}/{len(idx)}")

    ms = np.concatenate(mask_stats)
    dist = {f"p{p}": float(np.percentile(ms, p)) for p in (1, 5, 10, 25, 50, 75, 90, 99)}
    out = {"mask_distribution": dist, "n_clips": len(idx), "variants": {}}
    for k, v in acc.items():
        out["variants"][k] = {m_: float(np.nanmean(v[m_])) for m_ in
                              ("stoi", "pesq", "sisdr", "gs", "gn")}
    json.dump(out, open(a.out, "w"), indent=1)

    print("\n  PREDICTED MASK DISTRIBUTION (operating point, not invented)")
    print("   " + "  ".join(f"{k} {v:.3f}" for k, v in dist.items()))
    b = out["variants"]["__noisy__"]; a0 = out["variants"]["A0 baseline"]
    print(f"\n  {len(idx)} clips.  noisy: STOI {b['stoi']:.3f}  PESQ {b['pesq']:.3f}  "
          f"SI-SDR {b['sisdr']:+.2f}")
    print(f"  {'variant':<26}{'STOI':>7}{'PESQ':>7}{'SI-SDR':>8}"
          f"{'speech dB':>11}{'noise dB':>10}{'dSTOI':>8}")
    print("  " + "-" * 78)
    for k in sorted(out["variants"]):
        if k == "__noisy__": continue
        v = out["variants"][k]
        print(f"  {k:<26}{v['stoi']:7.3f}{v['pesq']:7.3f}{v['sisdr']:8.2f}"
              f"{20*np.log10(max(v['gs'],EPS)):11.2f}{20*np.log10(max(v['gn'],EPS)):10.2f}"
              f"{v['stoi']-a0['stoi']:+8.3f}")
