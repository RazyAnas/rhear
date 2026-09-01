"""Predictability: the quantity that decides how much loop delay you can afford.

If the reference is predictable D samples ahead, D samples of electrical delay
cost nothing -- the controller synthesises anti-noise for an event it has already
seen. This module computes the theoretical floor on residual power for a given
delay, which is the bound the ANC loop should approach and must not beat.
"""
import numpy as np
from scipy import linalg


def _acf(x, lags):
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x - x.mean(), nfft)
    r = np.fft.irfft(np.abs(X) ** 2, nfft)[:lags]
    return r / n


def prediction_floor_db(x, delay_samples, order=256):
    """Normalised MMSE (dB) of predicting x(n) from x(n-D), x(n-D-1), ...

    D = 0 gives -inf (trivial). Large D approaches 0 dB (no cancellation possible).
    This is the information-theoretic floor for a feedforward controller whose
    total loop delay is D samples.
    """
    D = int(round(delay_samples))
    lags = order + D + 2
    r = _acf(x, lags)
    if r[0] <= 0:
        return 0.0
    p = r[D:D + order]
    # R is symmetric Toeplitz, so solve it with Levinson-Durbin rather than
    # building the full matrix and calling a general solver.
    #
    # This is not a micro-optimisation. The general path is O(n^3)/3 -- at
    # order=256, 5.59 M operations every 16 ms, which is 350 MMAC/s on its
    # own and blows the whole ESP32-S3 budget (measured: 150% utilisation,
    # FAIL). Levinson-Durbin is O(n^2): 131 k operations, 8.2 MMAC/s, and the
    # budget passes at 65%. Same answer, 43x less work.
    #
    # Regularisation is unchanged: adding 1e-10*r[0] to the diagonal of a
    # Toeplitz matrix is exactly adding it to the first autocorrelation lag.
    c = r[:order].copy()
    c[0] += 1e-10 * r[0]
    try:
        a = linalg.solve_toeplitz(c, p)
    except Exception:
        # Levinson assumes the leading minors stay non-singular. If the
        # autocorrelation is degenerate it can fail where a general solver
        # would not, so fall back rather than return a wrong number.
        a = np.linalg.solve(linalg.toeplitz(c), p)
    mmse = r[0] - float(a @ p)
    return 10 * np.log10(max(mmse, 1e-12) / r[0])


def predictability(x, delay_samples, order=256):
    """H in [0,1]: 1 means fully predictable at this delay, 0 means not at all."""
    return float(np.clip(1.0 - 10 ** (prediction_floor_db(x, delay_samples, order) / 10.0), 0, 1))
