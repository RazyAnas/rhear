#!/usr/bin/env python3
"""E08 - how short can the L0 filter be before cancellation suffers?

This decides a purchase, not just a parameter.

L0 must run at 192 kHz: the codec's 38 us group delay is quoted at that rate,
and its own engine does biquads, not a per-sample LMS update, so the adaptive
loop runs on the processor.

Counting the loop in rhear/core/anc.py, FxNLMS costs 3L + M MACs per sample per
channel. Against Espressif's published ESP-DSP benchmarks (0.834 MAC/cycle for
int16 at 240 MHz = 200 MMAC/s per core):

    L = 128   172 MMAC/s stereo   L0 on core 0, L1+L2 on core 1   ESP32-S3, Rs 409
    L = 256   319 MMAC/s          fills both cores, no room for L1
    L = 512   614 MMAC/s          out of reach; needs an STM32H7, Rs 1000+

So a short filter is worth real money. But our own window rule says exploiting a
repeating noise needs a filter at least one period long, and at 192 kHz:

    50 Hz  -> 3840 samples        200 Hz -> 960
    100 Hz -> 1920                1500 Hz -> 128

L = 128 therefore covers periodicity only above ~1500 Hz. The question this
answers is what that costs in dB, per kind of noise -- not whether it is
theoretically lossy, which we already know it is.

Everything here is the existing FxNLMS. No architecture change.
"""
import os, sys, json
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from rhear.core.anc import fxnlms

FS = 192_000            # forced by the codec, see docstring
DUR = 1.5
N = int(FS * DUR)
C_AIR = 343.0
D_SOURCE = 1.4
D_REFMIC = 0.07
TAU_ELEC = 38e-6
TAU_SPK = 0.02 / C_AIR
M_SEC = 64              # secondary-path model length
GAMMA2 = 0.95


def tonal(rng, f0, nh=12):
    """A repeating engine/rotor tone: fundamental plus harmonics."""
    t = np.arange(N) / FS
    x = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
            for k in range(1, nh + 1))
    return x * (1.0 + 0.3 * np.sin(2 * np.pi * 1.7 * t))


def band(rng, lo, hi):
    """Broadband rumble between lo and hi Hz.

    SOS, not (b, a). At 192 kHz a 40 Hz corner is a normalised frequency of
    4.2e-4, and the transfer-function form loses so much precision there that
    a pole lands at radius 1.0055 -- outside the unit circle. The filter is
    then unstable and the output is NaN, which looked like the CANCELLER
    diverging until the signal itself was checked.
    """
    sos = signal.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band",
                        output="sos")
    return signal.sosfilt(sos, rng.standard_normal(N))


def coh_mix(src, g2, rng):
    """Reference at coherence g2. Uncorrelated part is a random-phase surrogate
    so the spectrum matches and g2 is flat across frequency -- adding white
    noise to a band-limited source silently leaves in-band coherence far above
    what was asked for."""
    S = np.fft.rfft(src)
    ph = rng.uniform(0, 2 * np.pi, len(S)); ph[0] = 0.0
    ind = np.fft.irfft(np.abs(S) * np.exp(1j * ph), n=len(src))
    ind = ind / (np.std(ind) + 1e-12) * np.std(src)
    return np.sqrt(g2) * src + np.sqrt(max(0.0, 1 - g2)) * ind


def secondary():
    n_d = int(round((TAU_ELEC + TAU_SPK) * FS))
    h = np.zeros(n_d + 24); h[n_d:] = np.exp(-np.arange(24) / 8.0)
    return h / np.linalg.norm(h)


def delay(x, tau):
    d = int(round(tau * FS))
    return np.concatenate([np.zeros(d), x])[:len(x)]


def nr_db(d, e, skip=0.5):
    i = int(skip * len(d))
    return 10 * np.log10(np.sum(d[i:] ** 2) / (np.sum(e[i:] ** 2) + 1e-20))


def mmacs(L, ch=2):
    return (3 * L + M_SEC) * ch * FS / 1e6


if __name__ == "__main__":
    rng = np.random.default_rng(4242)
    s = secondary()
    p_h = rng.standard_normal(64) * np.exp(-np.arange(64) / 21.0)
    p_h /= np.linalg.norm(p_h)
    tau_p = D_SOURCE / C_AIR
    tau_r = tau_p - D_REFMIC / C_AIR

    cases = [
        ("engine 50 Hz  (period 3840)", lambda r: tonal(r, 50)),
        ("rotor 200 Hz  (period 960)", lambda r: tonal(r, 200)),
        ("hull 40-1200 Hz broadband", lambda r: band(r, 40, 1200)),
        ("wideband 200-8000 Hz", lambda r: band(r, 200, 8000)),
    ]
    Ls = [128, 256, 384, 512]

    print("  L0 at %d kHz, stereo, coherence %.2f (bound %.1f dB)\n"
          % (FS / 1000, GAMMA2, -10 * np.log10(1 - GAMMA2)))
    print("  %-30s%s" % ("noise", "".join("%10s" % ("L=%d" % L) for L in Ls)))
    print("  " + "-" * (30 + 10 * len(Ls)))

    out = {}
    for name, gen in cases:
        src = gen(rng); src = src / np.std(src)
        ref = coh_mix(src, GAMMA2, rng)          # one realisation for the row
        d = signal.lfilter(p_h, [1.0], delay(src, tau_p))
        x = delay(ref, tau_r)
        row = {}
        cells = ""
        for L in Ls:
            # mu ladder, best STABLE result -- not the first that runs.
            # At 192 kHz a narrowband tonal reference makes the NLMS
            # autocorrelation ill-conditioned and mu=0.05 diverges outright
            # (measured: -64.8 dB, i.e. the canceller became the noise
            # source). A small leak regularises the same problem. E01 hit
            # this and the fix is the same one.
            best, best_mu = -np.inf, None
            for mu in (2e-2, 5e-3, 1e-3, 2e-4, 5e-5, 1e-5, 2e-6):
                r = fxnlms(x, d, s, s, L=L, mu=mu, leak=1e-6)
                if r.get("diverged"):
                    continue
                v = nr_db(d, r["e"])
                if np.isfinite(v) and v > best:
                    best, best_mu = v, mu
            row["L%d" % L] = best if np.isfinite(best) else float("nan")
            row["mu_L%d" % L] = best_mu
            cells += "%10.1f" % row["L%d" % L]
        out[name] = row
        print("  %-30s%s" % (name, cells))

    print("\n  chosen step size mu (best stable per cell):")
    for name in out:
        print("    %-30s%s" % (name, "".join(
            "%10s" % (("%.0e" % out[name]["mu_L%d" % L]) if out[name]["mu_L%d" % L]
                      else "none") for L in Ls)))

    print("\n  %-30s%s" % ("compute, stereo MMAC/s",
                           "".join("%10.0f" % mmacs(L) for L in Ls)))
    print("  %-30s%s" % ("fits ESP32-S3 (200/core)",
                         "".join("%10s" % ("yes" if mmacs(L) <= 200 else
                                           ("2 cores" if mmacs(L) <= 400 else "no"))
                                 for L in Ls)))
    print("\n  cost of shortening, dB lost vs L=512:")
    for name in out:
        base = out[name]["L512"]
        print("    %-30s%s" % (name, "".join("%10.1f" % (out[name]["L%d" % L] - base)
                                             for L in Ls)))

    os.makedirs("results", exist_ok=True)
    json.dump({"fs": FS, "gamma2": GAMMA2, "nr_db": out,
               "mmac_s": {("L%d" % L): mmacs(L) for L in Ls}},
              open("results/e08_filter_length.json", "w"), indent=1)
    print("\n  -> results/e08_filter_length.json")
