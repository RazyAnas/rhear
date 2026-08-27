"""Bearing estimation: audio (GCC-PHAT), gyro dead-reckoning, and fusion."""
import numpy as np
from ..sim.geometry import theta_from_itd, mic_spacing, C_AIR


def gcc_phat(a, b, fs, max_tau=None, f_lo=100.0, f_hi=4000.0):
    """Cross-correlation with phase transform. Returns (tau_seconds, confidence).

    Note: zero-padding the FFT does NOT upsample the correlation -- the IFFT grid
    is still 1/fs. Sub-sample resolution comes from parabolic interpolation around
    the peak instead. (Getting this wrong produced a clean factor-of-8 bearing
    error on the first run.)

    The PHAT weighting must be BAND-LIMITED to where the source actually has
    energy. Normalising |R| to 1 everywhere promotes empty bands -- numerical
    noise with random phase -- to full weight, and for a band-limited source that
    is most of the spectrum. Unbanded PHAT put the peak at zero lag for four of
    seven test azimuths on the first run.

    Confidence is the peak-to-sidelobe margin. It collapses when the two channels
    decorrelate, which is what wind and low SNR do, and is what gates the fallback.
    """
    n = len(a) + len(b)
    nfft = 1 << int(np.ceil(np.log2(n)))
    A = np.fft.rfft(a - a.mean(), nfft)
    B = np.fft.rfft(b - b.mean(), nfft)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12                       # the PHAT weighting
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    R[(freqs < f_lo) | (freqs > f_hi)] = 0.0     # band-limit it (see docstring)
    cc = np.fft.irfft(R, nfft)
    md = max(int(np.ceil(fs * (max_tau or mic_spacing() / C_AIR))) + 1, 1)
    cc = np.concatenate((cc[-md:], cc[:md + 1]))
    mag = np.abs(cc)
    k = int(np.argmax(mag))
    # parabolic interpolation for sub-sample resolution
    if 0 < k < len(mag) - 1:
        y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-20 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    rest = mag.copy()
    rest[max(0, k - 3):k + 4] = 0.0
    conf = float((mag[k] - rest.max()) / (mag[k] + 1e-20))
    tau = (k + delta - md) / float(fs)
    return tau, max(0.0, min(1.0, conf))


class Gyro:
    """MEMS rate gyro: true rate plus a constant bias and white noise."""

    def __init__(self, bias_dps=0.5, noise_dps=0.05, rng=None):
        self.bias = bias_dps
        self.noise = noise_dps
        self.rng = rng or np.random.default_rng(0)

    def read(self, true_rate_dps):
        return true_rate_dps + self.bias + self.noise * self.rng.standard_normal()


class BearingTracker:
    """Complementary filter. The gyro carries the fast component; audio corrects
    the slow drift, weighted by its own confidence."""

    def __init__(self, theta0, tau_c=0.5, conf_floor=0.15):
        self.theta = float(theta0)
        self.tau_c = tau_c
        self.conf_floor = conf_floor
        self.fallbacks = 0
        self.n = 0

    def step(self, dt, gyro_dps=None, theta_audio=None, conf=0.0, mode="fusion"):
        self.n += 1
        if mode == "audio":
            if conf >= self.conf_floor and theta_audio is not None:
                self.theta = theta_audio
            else:
                self.fallbacks += 1
            return self.theta
        if mode == "imu":                        # dead-reckoning only: drifts
            self.theta += gyro_dps * dt
            return self.theta
        # fusion
        if gyro_dps is not None:
            self.theta += gyro_dps * dt
        if theta_audio is not None and conf >= self.conf_floor:
            k = (dt / self.tau_c) * conf         # correction weighted by confidence
            err = theta_audio - self.theta
            self.theta += min(k, 1.0) * err
        else:
            self.fallbacks += 1
        return self.theta
