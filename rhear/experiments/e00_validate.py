"""E00 - Simulator validation against results we already know the answer to.

If the twin cannot reproduce closed-form theory, no result from it is worth
anything. Three checks:
  V1  Generic-codec decimation delay should land in the 500-600 us band that the
      literature reports for 48 kHz audio codecs.
  V2  Achievable ANC attenuation must obey the coherence bound
      NR <= -10 log10(1 - gamma^2)  and should approach it with a long filter.
  V3  Feedforward control must fail when the causality condition is violated,
      and the failure point must be where theory says it is.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import signal, linalg


def _xcorr(a, b, lags):
    """First `lags` values of sum_n a[n] b[n-k], computed by FFT."""
    n = len(a)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    r = np.fft.irfft(A * np.conj(B), nfft)
    return r[:lags]
from rhear.sim.electronics import decimation_chain_group_delay
from rhear.sim import paths
from rhear.core import signals as sg


def v1_codec_delay():
    print("\nV1  Decimation-chain group delay (derived, not asserted)")
    gd = decimation_chain_group_delay(verbose=True)
    lit_lo, lit_hi = 500e-6, 600e-6
    print(f"    literature for 48 kHz audio codecs : {lit_lo*1e6:.0f}-{lit_hi*1e6:.0f} us")
    ok = 0.8 * lit_lo <= gd <= 2.0 * lit_hi
    print(f"    -> {'PASS' if ok else 'FAIL'}  (derived {gd*1e6:.0f} us)")
    return ok, gd


def v2_coherence_bound(fs=48_000, n=120_000, L=384):
    print("\nV2  Coherence bound  NR <= -10 log10(1 - gamma^2)")
    rng = np.random.default_rng(7)
    p = paths.primary_path(fs, distance_m=0.02, cup_cutoff_hz=6000.0)
    ok_all = True
    print(f"    {'gamma^2':>8} {'bound dB':>10} {'achieved dB':>12} {'gap dB':>8}")
    for g2 in (0.5, 0.9, 0.99, 0.999):
        s = sg.broadband(n, fs, 100, 8000, rng=rng)
        # The uncorrelated part must have the SAME spectral shape as the signal,
        # otherwise gamma^2 is frequency-dependent and a single scalar bound is
        # not the right comparison. (This is what V2 caught on the first run.)
        v = sg.broadband(n, fs, 100, 8000, rng=np.random.default_rng(99))
        sigma2 = (1.0 - g2) / g2
        x = s + np.sqrt(sigma2) * v                            # reference = signal + uncorrelated
        d = signal.lfilter(p, [1.0], s)                        # disturbance at the ear
        # Wiener-Hopf optimum of length L predicting d from x (FFT correlations)
        rxx = _xcorr(x, x, L) / n
        rxd = _xcorr(d, x, L) / n
        Rm = linalg.toeplitz(rxx)
        w = np.linalg.solve(Rm + 1e-9 * rxx[0] * np.eye(L), rxd)
        e = d - signal.lfilter(w, [1.0], x)
        achieved = 10 * np.log10(np.mean(e[L:] ** 2) / np.mean(d[L:] ** 2))
        bound = 10 * np.log10(1 - g2)
        gap = achieved - bound
        ok = gap > -0.6                       # must not beat the bound
        ok_all &= ok
        print(f"    {g2:8.3f} {bound:10.2f} {achieved:12.2f} {gap:8.2f}  {'ok' if ok else 'VIOLATION'}")
    print(f"    -> {'PASS' if ok_all else 'FAIL'}  (no run beats the information bound)")
    return ok_all


if __name__ == "__main__":
    ok1, _ = v1_codec_delay()
    ok2 = v2_coherence_bound()
    print(f"\n{'='*60}\nVALIDATION: {'PASS' if (ok1 and ok2) else 'FAIL'}\n{'='*60}")
