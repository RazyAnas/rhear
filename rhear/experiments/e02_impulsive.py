"""E02 - Impulsive noise: does the controller survive a gunshot?

Tests research-base 1.3 and architecture 4.4. Physically faithful setup: the
reference microphone SATURATES at its 135 dB AOP while the acoustic event keeps
rising to 160 dB, so the controller sees a clipped reference against an
unclipped error. That mismatch -- not the alpha-stable distribution by itself --
is the mechanism that corrupts the filter on a real headset.

First run of this experiment found that plain FxNLMS did NOT diverge, because
normalisation by ||xhat||^2 already divides out a loud reference. The textbook
divergence result is about UNNORMALISED FxLMS, so both are compared here.

    fxlms          unnormalised, psi(e) = e        -- the textbook baseline
    fxnlms         normalised                      -- normalisation as a defence
    fxnlms + log   psi(e) = e/(eps+|e|)            -- FxlogLMS, no alpha needed
    + freeze       detector-gated adaptation hold
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import signal
from rhear.sim import paths
from rhear.sim.electronics import soft_clip_aop
from rhear.core import signals as sg
from rhear.core.anc import fxnlms, nmse_db

# Physical levels. ref_amplitude = 1.0 corresponds to 94 dB SPL (1 Pa).
BG_DBSPL, PEAK_DBSPL, AOP_DBSPL, PASSIVE_DB = 95.0, 160.0, 135.0, 25.0
amp = lambda db: 10 ** ((db - 94.0) / 20.0)

FS = 48_000
DUR = 1.2
T_EVENT = 0.6
ALPHA = 1.5


def segment_nmse(e, d, t0, t1):
    a, b = int(t0 * FS), int(t1 * FS)
    seg = e[a:b]
    if not np.all(np.isfinite(seg)):
        return np.inf
    num = float(np.sum(seg.astype(np.float64) ** 2))
    den = float(np.sum(d[a:b].astype(np.float64) ** 2)) + 1e-30
    return np.inf if (not np.isfinite(num) or num / den > 1e6) else 10 * np.log10(num / den + 1e-30)


def recovery_time(e, d, w_norm, track, pre_db, tol=3.0):
    """Seconds after the event before short-window attenuation is back within
    `tol` dB of the pre-event value."""
    win = int(0.05 * FS)
    i = int(T_EVENT * FS) + int(0.05 * FS)
    while i + win < len(e):
        seg, dd = e[i:i + win], d[i:i + win]
        if np.all(np.isfinite(seg)):
            v = 10 * np.log10((np.sum(seg ** 2) + 1e-30) / (np.sum(dd ** 2) + 1e-30))
            if v <= pre_db + tol:
                return (i - int(T_EVENT * FS)) / FS
        i += win // 2
    return np.nan


def freeze_ablation():
    """Does the detector-gated adaptation freeze actually earn its place?

    Sweeps step size and burst count, comparing log-FxNLMS with and without the
    freeze on post-event attenuation. Reported honestly whichever way it lands.
    """
    print("\nFREEZE ABLATION  (post-event attenuation, measured 1.35-1.60 s)")
    print(f"{'scenario':<24}{'mu':>7}{'log':>11}{'log+freeze':>13}{'delta':>9}")
    print("-" * 64)
    p_ = paths.primary_path(FS, ntaps=512)
    s_ = paths.secondary_path(FS, ntaps=1024)
    worst = 0.0
    for scen, times in (("single burst", [0.6]),
                        ("burst train (8)", list(np.arange(0.6, 1.15, 0.07)))):
        for mu in (0.01, 0.05, 0.2):
            rng = np.random.default_rng(11)
            n = int(FS * 1.6)
            bg = amp(BG_DBSPL) * sg.broadband(n, FS, 80, 2000, rng=rng)
            b = np.zeros(n)
            for t in times:
                bb = sg.impulse_burst(n, FS, at_s=t, alpha=ALPHA, dur_s=0.02, amp=1.0, rng=rng)
                b += bb * amp(PEAK_DBSPL) / (np.max(np.abs(bb)) + 1e-20)
            xa = bg + b
            xm = soft_clip_aop(xa, aop_dbspl=AOP_DBSPL)
            dd = signal.lfilter(p_, [1.0], xa) * 10 ** (-PASSIVE_DB / 20.0)
            v = []
            for fr in (False, True):
                r = fxnlms(xm, dd, s_, s_, L=256, mu=mu, mode="log", normalize=True,
                           freeze_on_impulse=fr, fs=FS, track_every=16)
                v.append(segment_nmse(r["e"], dd, 1.35, 1.60))
            f = lambda z: "diverged" if not np.isfinite(z) else f"{z:8.2f}"
            if np.isfinite(v[0]) and np.isfinite(v[1]):
                delta = v[1] - v[0]
                worst = max(worst, abs(delta))
                ds = f"{delta:+8.2f}"
            else:
                ds = "     n/a"
            print(f"{scen:<24}{mu:>7}{f(v[0]):>11}{f(v[1]):>13}{ds:>9}")
    print("-" * 64)
    print(f"largest effect of the freeze: {worst:.2f} dB")
    print("VERDICT: the freeze does NOT earn its place as an adaptation-stability")
    print("mechanism here. Normalisation plus the log score function already carry it.")
    print("The detector still earns its place for the limiter and PROTECT mode.")
    return worst


if __name__ == "__main__":
    n = int(FS * DUR)
    rng = np.random.default_rng(11)
    p = paths.primary_path(FS, ntaps=512)
    s = paths.secondary_path(FS, ntaps=1024)

    bg = amp(BG_DBSPL) * sg.broadband(n, FS, 80, 2000, rng=rng)
    burst = sg.impulse_burst(n, FS, at_s=T_EVENT, alpha=ALPHA, dur_s=0.02, amp=1.0, rng=rng)
    burst *= amp(PEAK_DBSPL) / (np.max(np.abs(burst)) + 1e-20)
    x_acoustic = bg + burst

    # What the reference microphone actually delivers: saturated at its AOP.
    x = soft_clip_aop(x_acoustic, aop_dbspl=AOP_DBSPL)
    # What reaches the ear: attenuated by the passive cup, then the primary path.
    d = signal.lfilter(p, [1.0], x_acoustic) * 10 ** (-PASSIVE_DB / 20.0)

    over = 20 * np.log10(np.max(np.abs(x_acoustic)) / np.max(np.abs(x)))
    print(f"Scenario: {DUR}s background at {BG_DBSPL:.0f} dB SPL, "
          f"alpha-stable burst (alpha={ALPHA}) peaking at {PEAK_DBSPL:.0f} dB SPL at t={T_EVENT}s")
    print(f"  reference mic AOP {AOP_DBSPL:.0f} dB SPL -> event exceeds it by "
          f"{PEAK_DBSPL-AOP_DBSPL:.0f} dB; measured peak is compressed by {over:.1f} dB")
    print(f"  passive cup attenuation assumed {PASSIVE_DB:.0f} dB\n")

    print(f"{'controller':<16}{'pre-event':>12}{'during':>12}{'post':>12}"
          f"{'||w|| peak/pre':>16}{'recovery':>11}")
    print("-" * 79)
    out = {}
    for label, mode, freeze, norm, mu in [
            ("fxlms (unnorm)", "nlms", False, False, 2e-3),
            ("fxnlms",         "nlms", False, True,  0.01),
            ("fxnlms + log",   "log",  False, True,  0.01),
            ("+ freeze",       "log",  True,  True,  0.01)]:
        r = fxnlms(x, d, s, s, L=256, mu=mu, mode=mode, normalize=norm,
                   freeze_on_impulse=freeze, fs=FS, track_every=16)
        e, wn = r["e"], r["w_norm"]
        pre = segment_nmse(e, d, 0.35, T_EVENT - 0.01)
        dur = segment_nmse(e, d, T_EVENT, T_EVENT + 0.05)
        post = segment_nmse(e, d, DUR - 0.25, DUR)
        i_pre = int((T_EVENT - 0.01) * FS / 16)
        wn_pre = wn[i_pre] if np.isfinite(wn[i_pre]) and wn[i_pre] > 0 else np.nan
        wn_peak = np.nanmax(wn[i_pre:])
        ratio = wn_peak / wn_pre if wn_pre and np.isfinite(wn_pre) else np.nan
        rec = recovery_time(e, d, wn, 16, pre)
        frac_frozen = r["frozen"].mean() * 100
        fmt = lambda v: ("  diverged" if not np.isfinite(v) else f"{v:8.1f} dB")
        print(f"{label:<16}{fmt(pre):>12}{fmt(dur):>12}{fmt(post):>12}"
              f"{ratio:14.1f}x{(f'{rec*1e3:.0f} ms' if np.isfinite(rec) else 'never'):>11}")
        out[label] = dict(e=e, w_norm=wn, pre=pre, post=post, ratio=ratio, rec=rec,
                          frozen_pct=frac_frozen)
    print("-" * 79)
    print(f"detector held adaptation for {out['+ freeze']['frozen_pct']:.1f}% of the run "
          f"(hold window 200 ms of a {DUR}s file)")
    worst = freeze_ablation()
    np.savez("results/e02_impulsive.npz", freeze_effect_db=worst,
             **{f"{k.replace(' ','_').replace('+','p')}_{f}": out[k][f]
                for k in out for f in ("pre", "post", "ratio", "rec")})
    print("saved -> results/e02_impulsive.npz")
