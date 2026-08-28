#!/usr/bin/env python3
"""PART C - quiet-speech forensics on the FROZEN checkpoint.

Read-only: loads baseline_frozen/model.pt and the fixed 300-clip test split.
Writes only into partC/. Measures first; draws no conclusion here.

NOTE ON PROVENANCE: the noise in this evaluation set is 100% SYNTHETICALLY
GENERATED. Findings about the SPEECH path are expected to transfer (the speech is
real LibriSpeech). Findings about specific noise types -- sirens above all, which
here are a single swept sinusoid -- are SYNTHETIC-ONLY and must not be
generalised to real recordings.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import GTCRNLite, N_FFT, HOP, FS
from train_interim import Pairs, stft, istft, si_sdr, DEV
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

SPEECH_TH_DB = -45.0      # clean-speech bin is "speech" if within this of its peak
EPS = 1e-10


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), EPS))


def analyse(model, win, x, y):
    """Return everything needed for one clip, on a common T-F grid."""
    xd = x[None].to(DEV)
    X = stft(xd, win)
    Y = stft(y[None].to(DEV), win)
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
    with torch.no_grad():
        mm, mp, spp = model(spec)
    mag = X.abs().transpose(1, 2).unsqueeze(1)
    ph = X.angle().transpose(1, 2).unsqueeze(1)
    Se = ((mm * mag) * torch.exp(1j * (ph + mp))).squeeze(1).transpose(1, 2)
    est = istft(Se, win, xd.shape[-1])

    m_pred = mm.squeeze(1).transpose(1, 2)[0].cpu().numpy()          # (F,T)
    m_orac = (Y.abs() / (X.abs() + 1e-8)).clamp(0, 1)[0].cpu().numpy()
    return dict(
        X=X[0].cpu().numpy(), Y=Y[0].cpu().numpy(), Se=Se[0].cpu().numpy(),
        est=est[0].cpu().numpy(), m_pred=m_pred, m_orac=m_orac,
        spp=spp.squeeze(1)[0].cpu().numpy())


def regions(Ymag):
    """Speech-active vs noise-only T-F bins, from the CLEAN spectrum."""
    peak = Ymag.max()
    sp = db(Ymag) > (db(peak) + SPEECH_TH_DB)
    return sp, ~sp


def frame_classes(Ymag, freqs):
    """Vowel-like vs consonant-like frames, from the clean speech only.

    Proxy, stated plainly: vowels are high-energy and harmonic (low spectral
    flatness); consonants (fricatives/stops) are lower-energy and noise-like
    (high flatness) with more high-frequency content. This is an approximation,
    not a phonetic labelling.
    """
    e = Ymag.sum(0)
    act = e > 0.1 * e.max()
    P = Ymag ** 2 + EPS
    flat = np.exp(np.mean(np.log(P), 0)) / (np.mean(P, 0) + EPS)
    hf = P[freqs > 2000].sum(0) / (P.sum(0) + EPS)
    vowel = act & (flat < np.median(flat[act]) if act.any() else False) & (hf < 0.5)
    cons = act & ~vowel
    return vowel, cons


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="baseline_frozen/model.pt")
    ap.add_argument("--n-plots", type=int, default=24)
    ap.add_argument("--out", default="partC")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    model = GTCRNLite().to(DEV)
    model.load_state_dict(torch.load(a.ckpt, map_location=DEV)); model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)
    print(f"  frozen ckpt {a.ckpt}  |  {len(ds)} clips  |  device {DEV}")

    per_clip, lvl_num, lvl_den = [], np.zeros(12), np.zeros(12)
    gains = dict(speech=[], noise=[], vowel=[], cons=[], lf=[], hf=[])
    # speech-level bins in dB relative to that clip's speech peak
    edges = np.linspace(-60, 0, 13)

    for i in range(len(ds)):
        x, y = ds[i]
        r = analyse(model, win, x, y)
        Xm, Ym = np.abs(r["X"]), np.abs(r["Y"])
        Em = np.abs(r["Se"])
        sp, nz = regions(Ym)
        # gain actually applied, per bin
        g = Em / (Xm + EPS)
        gains["speech"].append(float(np.mean(g[sp])) if sp.any() else np.nan)
        gains["noise"].append(float(np.mean(g[nz])) if nz.any() else np.nan)
        lo, hi = freqs < 1000, freqs >= 1000
        gains["lf"].append(float(np.mean(g[sp & lo[:, None]])) if (sp & lo[:, None]).any() else np.nan)
        gains["hf"].append(float(np.mean(g[sp & hi[:, None]])) if (sp & hi[:, None]).any() else np.nan)
        vw, cs = frame_classes(Ym, freqs)
        gains["vowel"].append(float(np.mean(g[:, vw][sp[:, vw]])) if vw.any() and sp[:, vw].any() else np.nan)
        gains["cons"].append(float(np.mean(g[:, cs][sp[:, cs]])) if cs.any() and sp[:, cs].any() else np.nan)

        # retention vs input speech level (the key curve): per speech bin,
        # how much of the CLEAN level survives, bucketed by that bin's level
        rel = db(Ym[sp]) - db(Ym.max())
        keep = np.abs(r["Se"])[sp] / (Ym[sp] + EPS)      # output vs CLEAN, not vs noisy
        b = np.clip(np.digitize(rel, edges) - 1, 0, 11)
        for k in range(12):
            m_ = b == k
            if m_.any():
                lvl_num[k] += keep[m_].sum(); lvl_den[k] += m_.sum()

        c = y.numpy().astype(np.float64); e = r["est"].astype(np.float64)
        n = x.numpy().astype(np.float64)
        row = dict(id=ds.rows[i]["id"], snr_db=ds.rows[i]["snr_db"],
                   stationarity=ds.rows[i]["mixture_stationarity"],
                   classes=[l["cls"] for l in ds.rows[i]["noise_layers"]],
                   g_speech=gains["speech"][-1], g_noise=gains["noise"][-1])
        try: row["stoi"] = float(stoi_fn(c, e, FS)); row["stoi_noisy"] = float(stoi_fn(c, n, FS))
        except Exception: row["stoi"] = row["stoi_noisy"] = np.nan
        try: row["pesq"] = float(pesq_fn(FS, c, e, "wb")); row["pesq_noisy"] = float(pesq_fn(FS, c, n, "wb"))
        except Exception: row["pesq"] = row["pesq_noisy"] = np.nan
        row["sisdr"] = float(si_sdr(torch.from_numpy(e)[None], torch.from_numpy(c)[None]))
        row["sisdr_noisy"] = float(si_sdr(torch.from_numpy(n)[None], torch.from_numpy(c)[None]))
        per_clip.append(row)
        if i % 60 == 0: print(f"    {i}/{len(ds)}")

    curve = (lvl_num / np.maximum(lvl_den, 1)).tolist()
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    g = {k: float(np.nanmean(v)) for k, v in gains.items()}
    summary = dict(
        PROVENANCE="noise is 100% SYNTHETICALLY GENERATED; speech is real LibriSpeech. "
                   "Speech-path findings expected to transfer; noise-type findings are synthetic-only.",
        checkpoint=a.ckpt, n_clips=len(per_clip),
        mean_gain=g,
        speech_gain_db=20 * np.log10(max(g["speech"], EPS)),
        noise_gain_db=20 * np.log10(max(g["noise"], EPS)),
        retention_curve=dict(level_db_rel_peak=centers, retention=curve),
        per_clip=per_clip)
    json.dump(summary, open(os.path.join(a.out, "forensics.json"), "w"), indent=1)

    print("\n  MEAN GAIN APPLIED (linear, and dB)")
    for k in ("speech", "noise", "vowel", "cons", "lf", "hf"):
        print(f"    {k:8s} {g[k]:6.3f}   {20*np.log10(max(g[k],EPS)):+7.2f} dB")
    print("\n  SPEECH RETENTION vs INPUT SPEECH LEVEL (output/clean, per T-F bin)")
    print(f"    {'level dB rel peak':>18}  retention")
    for c_, v in zip(centers, curve):
        if not np.isnan(v):
            bar = "#" * int(40 * min(v, 1.5))
            print(f"    {c_:18.0f}  {v:6.3f}  {bar}")
    print(f"\n  wrote {a.out}/forensics.json")
