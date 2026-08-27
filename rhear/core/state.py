"""Acoustic state features -- the input to L2 head A (architecture 5.2)."""
import numpy as np
from scipy import signal
from .predict import prediction_floor_db


def frame_features(x, fs, n_bands=12):
    """Compact descriptor of one analysis window. Cheap enough for 62.5 Hz."""
    if len(x) < 64 or np.std(x) < 1e-12:
        return np.zeros(n_bands + 5)
    xw = x - x.mean()
    X = np.abs(np.fft.rfft(xw * np.hanning(len(xw))))
    P = X ** 2 + 1e-20
    edges = np.geomspace(40.0, fs / 2 * 0.95, n_bands + 1)
    idx = np.clip((edges / (fs / 2) * (len(P) - 1)).astype(int), 0, len(P) - 1)
    bands = np.array([P[idx[i]:max(idx[i] + 1, idx[i + 1])].mean() for i in range(n_bands)])
    logb = np.log10(bands / (bands.sum() + 1e-20) + 1e-8)

    # periodicity: normalised autocorrelation peak away from lag 0
    r = np.fft.irfft(P)[:len(xw) // 2]
    r = r / (r[0] + 1e-20)
    lo = max(4, int(fs / 800))
    periodicity = float(np.max(r[lo:])) if len(r) > lo else 0.0

    # impulsiveness: kurtosis and crest factor
    s = np.std(xw) + 1e-20
    kurt = float(np.mean((xw / s) ** 4)) / 10.0
    crest = float(np.max(np.abs(xw)) / s) / 10.0

    # spectral flatness and centroid
    flat = float(np.exp(np.mean(np.log(P))) / (np.mean(P) + 1e-20))
    freqs = np.linspace(0, fs / 2, len(P))
    centroid = float((freqs * P).sum() / P.sum() / (fs / 2))

    return np.r_[logb, periodicity, kurt, crest, flat, centroid].astype(np.float32)
