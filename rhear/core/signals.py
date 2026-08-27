"""Noise generators: broadband, harmonic (rotor/engine), and alpha-stable impulses."""
import numpy as np
from scipy import signal


def broadband(n, fs, f_lo=80.0, f_hi=2000.0, rng=None):
    rng = rng or np.random.default_rng(1)
    x = rng.standard_normal(n)
    b, a = signal.butter(4, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
    x = signal.lfilter(b, a, x)
    return x / (np.std(x) + 1e-12)


def harmonic(n, fs, f0=200.0, n_harm=8, jitter=0.0, rng=None):
    """Rotor/engine-like: a harmonic comb, optionally with slow frequency drift."""
    rng = rng or np.random.default_rng(2)
    t = np.arange(n) / fs
    f0t = f0 * (1.0 + jitter * signal.lfilter(*signal.butter(2, 0.5 / (fs / 2)),
                                              rng.standard_normal(n))) if jitter else f0
    phase = 2 * np.pi * np.cumsum(np.full(n, f0t) if np.isscalar(f0t) else f0t) / fs
    x = sum(np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k for k in range(1, n_harm + 1))
    return x / (np.std(x) + 1e-12)


def alpha_stable(n, alpha=1.5, beta=0.0, scale=1.0, rng=None):
    """Symmetric alpha-stable via Chambers-Mallows-Stuck."""
    rng = rng or np.random.default_rng(3)
    U = rng.uniform(-np.pi / 2, np.pi / 2, n)
    W = rng.exponential(1.0, n)
    if abs(alpha - 1.0) < 1e-6:
        X = (2 / np.pi) * ((np.pi / 2 + beta * U) * np.tan(U)
                           - beta * np.log((np.pi / 2) * W * np.cos(U) / (np.pi / 2 + beta * U)))
    else:
        B = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        S = (1 + beta ** 2 * np.tan(np.pi * alpha / 2) ** 2) ** (1 / (2 * alpha))
        X = S * np.sin(alpha * (U + B)) / np.cos(U) ** (1 / alpha) * \
            (np.cos(U - alpha * (U + B)) / W) ** ((1 - alpha) / alpha)
    return scale * X


def impulse_burst(n, fs, at_s, alpha=1.5, dur_s=0.02, amp=50.0, rng=None):
    """A gunshot-like transient: alpha-stable excitation through a decaying envelope."""
    rng = rng or np.random.default_rng(4)
    x = np.zeros(n)
    i0 = int(at_s * fs)
    k = int(dur_s * fs)
    env = np.exp(-np.arange(k) / (0.15 * k))
    burst = alpha_stable(k, alpha=alpha, rng=rng) * env
    burst = np.clip(burst, -np.percentile(np.abs(burst), 99.9) * 3,
                    np.percentile(np.abs(burst), 99.9) * 3)
    x[i0:i0 + k] += amp * burst / (np.std(burst) + 1e-12)
    return x
