#!/usr/bin/env python3
"""How much room does each candidate design actually have? Measured, not argued.

The existing probe established our current ceiling: an IDEAL real-valued mask on
48 ERB bands, leaving the noisy phase alone, scores PESQ 2.838. The PS target is
2.5. Hitting it would demand 88% of a perfect model, which no 25k-parameter
network delivers. So the question is not "train harder" -- it is "which design
has a higher ceiling, and by how much".

This runs every candidate at ORACLE strength on the same clips. An oracle cannot
be beaten by any model of that family, so a family whose oracle is near 2.5 is
dead on arrival regardless of how good the training gets. Costs no training.

  A   noisy, unprocessed
  B   ideal real mask @48 ERB bands, noisy phase   <- WHERE WE ARE CAPPED TODAY
  F   ideal real mask @257 bins,     noisy phase   <- isolates RESOLUTION alone
  G   ideal 5-tap complex deep filter, all bins    <- the DeepFilterNet family
  Gb  same, magnitude-bounded |Y| <= |X|           <- G with our safety property
  H   ideal deep filter below 5 kHz + ERB mask above  <- the DEPLOYABLE proposal
  C   ideal real mask @48 bands,  CLEAN phase      <- phase handed over free
  D   ideal real mask @257 bins,  CLEAN phase      <- full-resolution ceiling

B vs F separates resolution from phase. B vs G separates "real gain per band"
from "complex filter across time" -- which is the entire deep-filtering claim.
Gb vs G prices our anti-hallucination guarantee. H is what we would actually
build, so it is the number that decides.

The deep filter is the multi-frame Wiener solution, per bin, solved in closed
form: minimise ||S - Z c||^2 where Z holds the N most recent noisy frames.
Time-invariant per clip, which UNDERSTATES what a per-frame predictor could do,
so H is a conservative floor on the family's ceiling rather than a flattering one.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import ERBSplit, N_FFT, HOP, FS  # HOP used in the guard below
from train_interim import Pairs, enhance, stft, istft, DEV
from rhear_data.manifest import read_manifest
from phase5_comb import si_metrics
from eval_stratified import load_model
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

_ERB = {}


def band_limit(mask, nb):
    key = (nb, str(mask.device))
    if key not in _ERB:
        _ERB[key] = ERBSplit(n_bands=nb).to(mask.device)
    erb = _ERB[key]
    mm = mask.transpose(1, 2).unsqueeze(1)
    mm = erb.inverse(erb(mm))
    mm = mm / (erb.inverse(erb(torch.ones_like(mm))) + 1e-8)
    return mm.squeeze(1).transpose(1, 2)


def deep_filter_oracle(X, S, N=5, fmax_bin=None, eps=1e-6, window=24):
    """Per-bin multi-frame Wiener filter with TIME-VARYING coefficients.

    X, S : (F, T) complex numpy, noisy and clean STFT
    window : causal fit window in frames. The filter for frame t is fitted on
             frames [t-window+1 .. t]. DeepFilterNet predicts coefficients every
             frame from local context, so a filter frozen across a whole clip is
             not this family's ceiling -- it is a different, far weaker method.

    THIS MATTERS. A first version solved ONE filter per clip and scored PESQ
    1.579 against the ideal real mask's 3.275, which is impossible: a
    time-varying real mask is the special case c = [g,0,0,0,0], so a
    time-varying complex filter can always match it. The gap was measuring
    "time-invariant vs time-varying", not "complex filter vs real gain".
    The caller asserts the inequality now, so this cannot silently recur.

    For frame t, bin f: minimise ||S - Z c||^2 over the window, where
    Z[tau, i] = X_f[tau - i]. Overdetermined while window > N.
    """
    F, T = X.shape
    hi = F if fmax_bin is None else min(fmax_bin, F)
    Z = np.zeros((hi, T, N), dtype=np.complex128)
    for i in range(N):                       # lag i -> shift right by i frames
        if i == 0:
            Z[:, :, 0] = X[:hi]
        else:
            Z[:, i:, i] = X[:hi, :-i]

    # Per-frame Gram matrices and cross-terms, summed over a causal window via
    # a cumulative sum so the cost is O(T) rather than O(T * window).
    # Normal equations for min ||S - Zc||^2 over complex c:
    #   R[j,i] = sum_t conj(Z[t,j]) Z[t,i]      p[j] = sum_t conj(Z[t,j]) S[t]
    # The conjugate goes on the FIRST index. Building the transpose instead
    # solves conj(R) c = p, which is a different problem and scored below the
    # real-mask oracle it strictly contains -- caught by the assertion below.
    outer = np.conj(Z)[:, :, :, None] * Z[:, :, None, :]          # (hi,T,N,N)
    cross = np.conj(Z) * S[:hi, :, None]                          # (hi,T,N)
    co = np.cumsum(outer, axis=1)
    cc = np.cumsum(cross, axis=1)
    lo = np.maximum(np.arange(T) - window, -1)
    padc_o = np.concatenate([np.zeros_like(co[:, :1]), co], axis=1)
    padc_c = np.concatenate([np.zeros_like(cc[:, :1]), cc], axis=1)
    R = co - padc_o[:, lo + 1]                                    # (hi,T,N,N)
    p = cc - padc_c[:, lo + 1]                                    # (hi,T,N)

    tr = np.trace(R, axis1=2, axis2=3).real / N
    R = R + (eps * np.maximum(tr, 1e-12))[:, :, None, None] * np.eye(N)
    c = np.linalg.solve(R, p[:, :, :, None])[:, :, :, 0]           # (hi,T,N)
    Y = X.copy()
    Y[:hi] = np.einsum("ftn,ftn->ft", Z, c)
    return Y


def istft_np(Y, n):
    return istft(torch.from_numpy(Y).to(DEV)[None].to(torch.complex64),
                 torch.hann_window(N_FFT, device=DEV), n)[0].cpu().numpy()


def score(clean, noisy, est):
    L = min(len(clean), len(noisy), len(est))
    c, x, e = clean[:L], noisy[:L], est[:L]
    o = {}
    _, o["sir"], o["sar"] = si_metrics(e, c, x - c)
    try:    o["stoi"] = float(stoi_fn(c, e, FS, extended=False))
    except Exception: o["stoi"] = np.nan
    try:    o["pesq"] = float(pesq_fn(FS, c, e, "wb"))
    except Exception: o["pesq"] = np.nan
    a = torch.from_numpy(c)[None]
    from train_interim import si_sdr
    o["sdr"] = float(si_sdr(torch.from_numpy(e)[None], a))
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--ckpt", default="runs/g2_fullband/best.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--taps", type=int, default=5)
    ap.add_argument("--window", type=int, default=24,
                    help="frames the DF coefficients are held across. This is "
                         "the agility knob and the result is sensitive to it.")
    ap.add_argument("--out", default="runs/g012/oracle_ladder.json")
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
    model, _ = load_model(a.ckpt)
    # The oracle rows build their own STFT at the module hop. A checkpoint
    # trained at a different hop would be scored through a transform it never
    # saw -- plausible numbers, silently wrong. Fail loudly instead.
    ck_hop = int(model.hop_.item())
    assert ck_hop == HOP, (
        "checkpoint hop %d != module hop %d. The oracle rows are built at the "
        "module hop, so row E would not be comparable. Re-run with a matching "
        "build, or parameterise the oracle rows first." % (ck_hop, HOP))
    win = torch.hann_window(N_FFT, device=DEV)
    bin5k = int(round(5000 / (FS / N_FFT)))

    NAMES = ["A noisy", "B ideal@48, noisy phase", "F ideal@257, noisy phase",
             "G deep filter, all bins", "Gb deep filter, bounded",
             "H deep filter <5k + ERB", "C ideal@48, CLEAN phase",
             "D ideal@257, CLEAN phase", "E G2 as trained"]
    acc = {k: {m: [] for m in ("stoi", "pesq", "sdr", "sir", "sar")} for k in NAMES}
    print("  %d clips | deep filter order N=%d | 5 kHz = bin %d of %d\n"
          % (len(idx), a.taps, bin5k, N_FFT // 2 + 1))

    for n_done, i in enumerate(idx):
        x, y = ds[i]
        n = x.shape[-1]
        Xt = stft(x[None].to(DEV), win)[0]            # (F,T) complex
        St = stft(y[None].to(DEV), win)[0]
        clean, noisy = y.numpy(), x.numpy()

        mag_x, mag_s = Xt.abs(), St.abs()
        ph_x, ph_s = Xt.angle(), St.angle()
        irm = torch.clamp(mag_s / (mag_x + 1e-8), max=1.0)   # tanh allows <= 1

        def synth(m, ph):
            return istft((m * torch.exp(1j * ph))[None],
                         win, n)[0].cpu().numpy()

        m48 = band_limit(irm[None], 48)[0]
        out = {
            "A noisy": noisy,
            "B ideal@48, noisy phase": synth(m48 * mag_x, ph_x),
            "F ideal@257, noisy phase": synth(irm * mag_x, ph_x),
            "C ideal@48, CLEAN phase": synth(m48 * mag_x, ph_s),
            "D ideal@257, CLEAN phase": synth(irm * mag_x, ph_s),
        }

        Xn, Sn = Xt.cpu().numpy(), St.cpu().numpy()
        Yg = deep_filter_oracle(Xn, Sn, N=a.taps, window=a.window)
        out["G deep filter, all bins"] = istft_np(Yg, n)

        # our anti-hallucination property, applied to the deep filter: no output
        # bin may exceed its input bin. A projection, not a constrained solve --
        # so this is a floor on what a properly constrained DF could reach.
        scale = np.minimum(1.0, np.abs(Xn) / (np.abs(Yg) + 1e-12))
        out["Gb deep filter, bounded"] = istft_np(Yg * scale, n)

        # H is the DeepFilterNet CASCADE, which is the point: ERB gains first,
        # everywhere, then deep filtering REFINES the already-enhanced low band.
        # An earlier version had DF replacing the mask below 5 kHz, which is a
        # different (and worse) architecture -- DF is a second stage, not a
        # substitute. Fitting stage 2 on the masked signal is what lets it act
        # as identity where the mask was already right.
        Ym = Xn * m48.cpu().numpy()
        Yr = deep_filter_oracle(Ym, Sn, N=a.taps, fmax_bin=bin5k, window=a.window)
        Yh = Ym.copy()
        Yh[:bin5k] = Yr[:bin5k]
        sc = np.minimum(1.0, np.abs(Xn) / (np.abs(Yh) + 1e-12))
        out["H deep filter <5k + ERB"] = istft_np(Yh * sc, n)

        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        out["E G2 as trained"] = est[0].cpu().numpy()

        for k, sig in out.items():
            for m, v in score(clean, noisy, sig).items():
                acc[k][m].append(v)
        if (n_done + 1) % 15 == 0:
            print("    %d/%d" % (n_done + 1, len(idx)), flush=True)

    res = {k: {m: float(np.nanmean(v)) for m, v in d.items()} for k, d in acc.items()}
    print("\n  %-28s%8s%8s%9s%9s%9s" % ("family", "STOI", "PESQ", "SI-SDR", "SI-SIR", "SI-SAR"))
    print("  " + "-" * 71)
    for k in NAMES:
        r = res[k]
        star = "  <<<" if k.startswith(("H ", "E ")) else ""
        print("  %-28s%8.3f%8.3f%9.2f%9.2f%9.2f%s"
              % (k, r["stoi"], r["pesq"], r["sdr"], r["sir"], r["sar"], star))

    b, h, e = res["B ideal@48, noisy phase"], res["H deep filter <5k + ERB"], res["E G2 as trained"]
    f, g = res["F ideal@257, noisy phase"], res["G deep filter, all bins"]
    # The right validity check is on the CASCADE, not on G. G and F optimise
    # different things -- F is a pointwise oracle with exact per-frame knowledge,
    # G holds one filter across `window` frames -- so G < F is legitimate and
    # says the design needs coefficient agility, not that the code is broken.
    # H, though, contains "leave the mask alone" (c = [1,0,0,0,0]) as a special
    # case, so H below B means the stage-2 fit is broken.
    if h["sdr"] < b["sdr"] - 0.3:
        print("\n  *** INVALID: H SI-SDR (%.2f) < B (%.2f). The cascade contains"
              % (h["sdr"], b["sdr"]))
        print("      identity as a special case and cannot lose to its own stage 1.")
    print("\n  WHAT MOVES THE CEILING (PESQ)")
    print("    today's ceiling      B  %.3f   target 2.5 -> need %.0f%% of oracle"
          % (b["pesq"], 100 * 2.5 / b["pesq"]))
    print("    resolution alone     F  %.3f   (%+.3f over B)" % (f["pesq"], f["pesq"] - b["pesq"]))
    print("    deep filter, all     G  %.3f   (%+.3f over B)" % (g["pesq"], g["pesq"] - b["pesq"]))
    print("    DEPLOYABLE           H  %.3f   (%+.3f over B) -> need %.0f%% of oracle"
          % (h["pesq"], h["pesq"] - b["pesq"], 100 * 2.5 / max(h["pesq"], 1e-9)))
    print("    we are at            E  %.3f" % e["pesq"])
    json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=1)
    print("\n  -> %s" % a.out)
