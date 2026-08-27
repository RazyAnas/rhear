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
    R = linalg.toeplitz(r[:order])
    p = r[D:D + order]
    a = np.linalg.solve(R + 1e-10 * r[0] * np.eye(order), p)
    mmse = r[0] - float(a @ p)
    return 10 * np.log10(max(mmse, 1e-12) / r[0])


def predictability(x, delay_samples, order=256):
    """H in [0,1]: 1 means fully predictable at this delay, 0 means not at all."""
    return float(np.clip(1.0 - 10 ** (prediction_floor_db(x, delay_samples, order) / 10.0), 0, 1))
