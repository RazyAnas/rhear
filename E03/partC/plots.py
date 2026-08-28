#!/usr/bin/env python3
"""PART C figures. Identical axes and a FIXED common dB/colour scale everywhere."""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
sys.path.insert(0, os.path.join(HERE, "partC"))
from gtcrn_lite import N_FFT, HOP, FS, GTCRNLite
from train_interim import Pairs, DEV
from rhear_data.manifest import read_manifest
from forensics import analyse, db

VMIN, VMAX = -80.0, 0.0          # fixed dB scale for EVERY spectrogram
WMAX = 1.0                       # fixed waveform scale


def spec_panel(ax, M, ref, title, freqs, T):
    S = db(M) - db(ref)
    im = ax.pcolormesh(np.arange(M.shape[1]) * HOP / FS, freqs / 1000, S,
                       vmin=VMIN, vmax=VMAX, shading="auto", cmap="magma")
    ax.set_title(title, fontsize=7); ax.set_ylim(0, 8)
    ax.tick_params(labelsize=6)
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="baseline_frozen/model.pt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="partC/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    model = GTCRNLite().to(DEV)
    model.load_state_dict(torch.load(a.ckpt, map_location=DEV)); model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    freqs = np.linspace(0, FS / 2, N_FFT // 2 + 1)

    snr = np.array([r["snr_db"] for r in ds.rows])
    picks = np.argsort(snr)[np.linspace(0, len(snr) - 1, a.n).astype(int)]
    made = []
    for j, i in enumerate(picks):
        x, y = ds[int(i)]
        r = analyse(model, win, x, y)
        Xm, Ym, Em = np.abs(r["X"]), np.abs(r["Y"]), np.abs(r["Se"])
        ref = max(Ym.max(), Xm.max())            # ONE reference for all panels
        t = np.arange(len(x)) / FS
        fig, ax = plt.subplots(4, 2, figsize=(11, 9))
        for k, (sig, lbl, col) in enumerate((
                (y.numpy(), "clean", "#2b8a3e"), (x.numpy(), "noisy", "#c92a2a"),
                (r["est"], "enhanced", "#1971c2"))):
            ax[0][0].plot(t, sig + 0, lw=.4, color=col, alpha=.85, label=lbl)
        ax[0][0].set_ylim(-WMAX, WMAX); ax[0][0].legend(fontsize=6, loc="upper right")
        ax[0][0].set_title("waveforms (identical scale)", fontsize=7)
        ax[0][0].tick_params(labelsize=6)
        rr = ds.rows[int(i)]
        ax[0][1].axis("off")
        ax[0][1].text(0, .5, f"{rr['id']}\nSNR {rr['snr_db']:+.1f} dB\n"
                             f"{'+'.join(l['cls'] for l in rr['noise_layers'])}\n"
                             f"{rr['mixture_stationarity']}\n"
                             f"speaker {rr['speaker']}\n\n"
                             f"dB scale fixed {VMIN}..{VMAX}\nSYNTHETIC NOISE",
                      fontsize=7, va="center", family="monospace")
        im = spec_panel(ax[1][0], Ym, ref, "clean spectrogram", freqs, t)
        spec_panel(ax[1][1], Xm, ref, "noisy spectrogram", freqs, t)
        spec_panel(ax[2][0], Em, ref, "ENHANCED spectrogram", freqs, t)
        spec_panel(ax[2][1], np.abs(Xm - Em), ref, "removed (noisy - enhanced)", freqs, t)
        for axx, M, ttl in ((ax[3][0], r["m_pred"], "PREDICTED magnitude mask"),
                            (ax[3][1], r["m_orac"], "ORACLE magnitude mask")):
            axx.pcolormesh(np.arange(M.shape[1]) * HOP / FS, freqs / 1000, M,
                           vmin=0, vmax=1, shading="auto", cmap="viridis")
            axx.set_title(ttl + "  (0..1 fixed)", fontsize=7); axx.set_ylim(0, 8)
            axx.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax[1:3, :].ravel().tolist(), shrink=.6, label="dB")
        f = os.path.join(a.out, f"clip{j:02d}_snr{rr['snr_db']:+05.1f}.png")
        fig.savefig(f, dpi=85, bbox_inches="tight"); plt.close(fig)
        made.append(os.path.basename(f))
        if j % 8 == 0: print(f"    {j}/{a.n}")

    # summary figure: retention curve + gain vs SNR
    d = json.load(open("partC/forensics.json"))
    pc = d["per_clip"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    c = d["retention_curve"]
    ax[0].plot(c["level_db_rel_peak"], c["retention"], "o-", color="#1971c2")
    ax[0].axhline(1, ls="--", c="#888", lw=.8)
    ax[0].set_xlabel("input speech level, dB rel. clip peak"); ax[0].set_ylabel("retention (out/clean)")
    ax[0].set_title("speech retention vs input level"); ax[0].grid(alpha=.3)
    s = np.array([r["snr_db"] for r in pc])
    gs = 20 * np.log10(np.array([r["g_speech"] for r in pc]))
    gn = 20 * np.log10(np.array([r["g_noise"] for r in pc]))
    ax[1].scatter(s, gs, s=8, c="#2b8a3e", label="speech gain")
    ax[1].scatter(s, gn, s=8, c="#c92a2a", label="noise gain")
    ax[1].set_xlabel("input SNR (dB)"); ax[1].set_ylabel("applied gain (dB)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_title("gain vs input SNR")
    st = np.array([r["stoi"] - r["stoi_noisy"] for r in pc])
    ax[2].scatter(s, st, s=8, c="#1971c2"); ax[2].axhline(0, ls="--", c="#888", lw=.8)
    ax[2].set_xlabel("input SNR (dB)"); ax[2].set_ylabel("delta STOI")
    ax[2].set_title("STOI change vs input SNR"); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "summary.png"), dpi=110)
    print(f"  {len(made)} clip figures + summary.png -> {a.out}")
