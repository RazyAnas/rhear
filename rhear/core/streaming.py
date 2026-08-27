"""Continuous signal generators with persistent state, for the live twin."""
import numpy as np
from scipy import signal


class StreamingHarmonic:
    def __init__(self, fs, f0, n_harm, rng):
        self.fs, self.f0, self.k = fs, f0, n_harm
        self.phase = 0.0
        self.ph = rng.uniform(0, 2 * np.pi, n_harm)

    def block(self, n):
        t = (np.arange(n) + 1) / self.fs
        ph = self.phase + 2 * np.pi * self.f0 * t
        self.phase = ph[-1]
        x = sum(np.sin(k * ph + self.ph[k - 1]) / k for k in range(1, self.k + 1))
        return x / (np.std(x) + 1e-9)


class StreamingBand:
    def __init__(self, fs, f_lo, f_hi, rng, order=4):
        self.b, self.a = signal.butter(order, [f_lo / (fs / 2), f_hi / (fs / 2)],
                                       btype="band")
        self.zi = signal.lfilter_zi(self.b, self.a) * 0.0
        self.rng = rng
        self._g = None

    def block(self, n):
        u = self.rng.standard_normal(n)
        y, self.zi = signal.lfilter(self.b, self.a, u, zi=self.zi)
        if self._g is None:
            self._g = 1.0 / (np.std(y) + 1e-9)
        return y * self._g


class StreamingDelayLine:
    """Fractional, time-varying delay -- how a moving source or a rotating head
    is rendered sample by sample."""
    def __init__(self, maxlen):
        self.buf = np.zeros(int(maxlen))
        self.n = 0

    def push_block(self, x):
        k = len(x)
        self.buf = np.roll(self.buf, -k)
        self.buf[-k:] = x
        self.n += k

    def read(self, delay_samples):
        """delay_samples: array, one per sample of the most recent block."""
        k = len(delay_samples)
        idx = len(self.buf) - k + np.arange(k) - delay_samples
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        i0 = np.clip(i0, 0, len(self.buf) - 2)
        return (1 - fr) * self.buf[i0] + fr * self.buf[i0 + 1]
