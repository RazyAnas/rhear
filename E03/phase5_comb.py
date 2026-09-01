#!/usr/bin/env python3
"""H9 - comb POST-FILTER on a frozen checkpoint. No retraining.

Why. The exact mask, read off gtcrn_lite.forward + train_interim.enhance:

    S_hat(t,f) = tanh(|m(t,f)|) * |X(t,f)| * exp(j*(angle X(t,f) + angle m(t,f)))

with m emitted at 48 ERB bands and expanded to 257 bins. Measured on 40 E_def
clips: 0 of 10.3M bins exceed |X|, median gain 0.090, 51.6% of bins attenuated
by more than 20 dB.

So the mask can attenuate any bin and rotate any phase, but it cannot vary its
gain faster than the 48-band grid. Measured band widths: above ~1250 Hz a
single band spans more than a 120 Hz harmonic spacing, reaching 594 Hz (about
5 harmonics) in the top band. Inside such a band the voiced harmonics and the
noise between them share one gain, so within-band SNR cannot be improved. That
residual is what PESQ punishes.

A comb filter aligned to the pitch period has resolution set by that period
rather than by the band grid, so it can attenuate between harmonics. Applied
here in the STFT domain, which is exact for a constant lag within a frame:

    H_t(k) = (1 - b_t) + b_t * exp(-j*2*pi*k*T_t/N_FFT)

|H| = 1 at multiples of F0 = FS/T and dips to |1-2b| between them. For
b in [0, 0.5], |H| <= 1 everywhere, so the post-filter attenuates and the
"never exceeds the input" property of the mask is preserved -- which this
script TESTS on every bin rather than asserting.

Applied only above F_MIN_HZ, because below it the band grid already resolves
harmonics and combing there would only risk damage.
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import N_FFT, HOP, FS, GTCRNLite
from train_interim import Pairs, enhance, istft, DEV
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn

F_MIN_HZ = 1250.0        # below this the 48-band grid already resolves harmonics
F0_MIN, F0_MAX = 60.0, 400.0
BETA_MAX = 0.5           # keeps |H| <= 1
R_FLOOR = 0.35           # below this periodicity, do not comb at all


def f0_track(x, n_frames, hop=HOP, win=N_FFT):
    """Per-frame pitch lag and normalised autocorrelation, from the ENHANCED
    signal (cleaner than the noisy input, so the lag is more reliable).
    Plain autocorrelation via FFT -- no external pitch tracker needed."""
    lo, hi = int(FS / F0_MAX), int(FS / F0_MIN)
    T = np.zeros(n_frames, dtype=int)
    R = np.zeros(n_frames)
    xp = np.concatenate([np.zeros(win // 2), x, np.zeros(win)])
    nfft = 1 << int(np.ceil(np.log2(2 * win)))
    for t in range(n_frames):
        s = xp[t * hop: t * hop + win]
        if len(s) < win:
            break
        s = s - s.mean()
        e0 = np.dot(s, s)
        if e0 < 1e-10:
            continue
        S = np.fft.rfft(s, nfft)
        ac = np.fft.irfft(np.abs(S) ** 2, nfft)[:hi + 1]
        seg = ac[lo:hi + 1]
        if seg.size == 0:
            continue
        k = int(np.argmax(seg)) + lo
        T[t], R[t] = k, float(ac[k] / (ac[0] + 1e-12))
    return T, R


def comb_postfilter(S, x_enh, f_min=F_MIN_HZ, beta_max=BETA_MAX,
                    r_floor=R_FLOOR):
    """S: (F,T) complex STFT of the enhanced signal. Returns combed STFT."""
    F, nT = S.shape
    T_lag, R = f0_track(x_enh, nT)
    k = np.arange(F)
    kmin = int(np.ceil(f_min * N_FFT / FS))
    out = S.copy()
    for t in range(nT):
        if T_lag[t] == 0 or R[t] < r_floor:
            continue
        b = beta_max * min(1.0, (R[t] - r_floor) / (1.0 - r_floor))
        H = (1 - b) + b * np.exp(-2j * np.pi * k * T_lag[t] / N_FFT)
        out[kmin:, t] = S[kmin:, t] * H[kmin:]
    return out, T_lag, R


def si_metrics(est, ref, noise):
    """Scale-invariant SDR / SIR / SAR.

    SI-SDR alone cannot separate 'removed noise' from 'damaged speech'.
    Projecting onto span{clean, noise} splits the error into interference
    (what SIR measures -- noise suppression) and artefacts (what SAR
    measures -- distortion the model invented).
    """
    L = min(len(est), len(ref), len(noise))
    est, ref, noise = est[:L], ref[:L], noise[:L]
    est = est - est.mean(); ref = ref - ref.mean(); noise = noise - noise.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_t = a * ref
    B = np.stack([ref, noise], 1)
    G = B.T @ B + 1e-10 * np.eye(2)
    P = B @ np.linalg.solve(G, B.T @ est)
    e_int = P - s_t
    e_art = est - P
    d = lambda n, dn: 10 * np.log10((np.sum(n ** 2) + 1e-12) /
                                    (np.sum(dn ** 2) + 1e-12))
    return (d(s_t, est - s_t), d(s_t, e_int), d(P, e_art))


def score(clean, sig):
    try:
        s = float(stoi_fn(clean, sig, FS, extended=False))
    except Exception:
        s = np.nan
    try:
        p = float(pesq_fn(FS, clean, sig, "wb"))
    except Exception:
        p = np.nan
    return s, p


def run(data, ckpt, tag, n=None):
    hdr, rows = read_manifest(os.path.join(data, "manifest.jsonl"))
    ds = Pairs(data, "test", rows, cache=False)
    meta = [r for r in rows if r["split"] == "test"]
    model = GTCRNLite().to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    sel = range(len(ds)) if n is None else \
        np.linspace(0, len(ds) - 1, n).astype(int)

    res, viol, bins, t_comb, t_frames = [], 0, 0, 0.0, 0
    for i in sel:
        x, y = ds[int(i)]
        c = y.numpy().astype(np.float64)
        nz = x.numpy().astype(np.float64)
        with torch.no_grad():
            Se, est, X, _ = enhance(model, x[None].to(DEV), win)
        e = est[0].cpu().numpy().astype(np.float64)
        Snp = Se[0].cpu().numpy()
        Xnp = X[0].cpu().numpy()

        t0 = time.perf_counter()
        Sc, T_lag, R = comb_postfilter(Snp, e)
        t_comb += time.perf_counter() - t0
        t_frames += Snp.shape[1]

        ec = istft(torch.from_numpy(Sc)[None].to(DEV), win,
                   x.shape[-1])[0].cpu().numpy().astype(np.float64)
        # content-preservation test, measured not assumed
        viol += int(np.sum(np.abs(Sc) > np.abs(Xnp) * 1.0001 + 1e-9))
        bins += Sc.size

        L = min(len(c), len(nz), len(e), len(ec))
        c, nz, e, ec = c[:L], nz[:L], e[:L], ec[:L]
        noise = nz - c
        r = dict(meta[int(i)])
        for lbl, sig in (("noisy", nz), ("h3", e), ("comb", ec)):
            st, pq = score(c, sig)
            sd, si, sa = si_metrics(sig, c, noise)
            r.update({f"{lbl}_stoi": st, f"{lbl}_pesq": pq,
                      f"{lbl}_sdr": sd, f"{lbl}_sir": si, f"{lbl}_sar": sa})
        r["voiced_frac"] = float(np.mean(R >= R_FLOOR))
        res.append(r)
    return res, viol, bins, t_comb, t_frames


def agg(rows, lbl):
    f = lambda k: float(np.nanmean([r[f"{lbl}_{k}"] for r in rows]))
    return {k: f(k) for k in ("stoi", "pesq", "sdr", "sir", "sar")}


def line(name, a):
    return ("  %-22s STOI %.3f  PESQ %.3f  SI-SDR %+6.2f  "
            "SI-SIR %+6.2f  SI-SAR %+6.2f"
            % (name, a["stoi"], a["pesq"], a["sdr"], a["sir"], a["sar"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/h9_frozen/h3_snapshot.pt")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="runs/h9_frozen/h9_results.json")
    a = ap.parse_args()

    allout = {}
    for d in a.sets:
        tag = os.path.basename(d.rstrip("/"))
        res, viol, bins, tc, tf = run(d, a.ckpt, tag, a.n)
        print("\n=== %s   (%d clips, ckpt %s) ===" % (tag, len(res), a.ckpt))
        A, B, C = agg(res, "noisy"), agg(res, "h3"), agg(res, "comb")
        print(line("noisy input", A))
        print(line("H3", B))
        print(line("H3 + comb", C))
        print("  %-22s dSTOI %+.4f  dPESQ %+.4f  dSI-SDR %+.2f  "
              "dSI-SIR %+.2f  dSI-SAR %+.2f"
              % ("comb - H3", C["stoi"] - B["stoi"], C["pesq"] - B["pesq"],
                 C["sdr"] - B["sdr"], C["sir"] - B["sir"], C["sar"] - B["sar"]))
        print("  content-preservation: %d of %d bins exceed |X|  (%.4f%%)"
              % (viol, bins, 100.0 * viol / max(1, bins)))
        sub = {}
        for lo, hi, nm in [(-99, 0, "low SNR (<0 dB)"), (0, 15, "mid (0-15 dB)"),
                           (15, 99, "high SNR (>15 dB)")]:
            s = [r for r in res if lo <= r.get("snr_db", -999) < hi]
            if not s:
                continue
            b, c = agg(s, "h3"), agg(s, "comb")
            sub[nm] = {"n": len(s), "h3": b, "comb": c}
            print("    %-18s n=%3d  dPESQ %+.3f  dSTOI %+.4f  dSI-SAR %+.2f"
                  % (nm, len(s), c["pesq"] - b["pesq"], c["stoi"] - b["stoi"],
                     c["sar"] - b["sar"]))
        allout[tag] = {"noisy": A, "h3": B, "comb": C, "by_snr": sub,
                       "violations": viol, "bins": bins,
                       "comb_ms_per_frame": 1000.0 * tc / max(1, tf),
                       "voiced_frac": float(np.mean([r["voiced_frac"]
                                                     for r in res]))}
    json.dump(allout, open(a.out, "w"), indent=1)
    print("\n  -> %s" % a.out)
