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


def coherence_confidence(a, b, fs, f_lo=120.0, f_hi=1500.0, nseg=6):
    """Magnitude-squared coherence between the two references, band-averaged.

    Replaces the peak-to-sidelobe margin, which is WRONG for this application:
    for a tonal or harmonic source the GCC surface is periodic, so the sidelobe
    margin collapses even when the bearing estimate is perfect. Measured in E05,
    every arm reported 100% fallback on engine and rotor scenes.

    Coherence asks the right question -- "are these two channels linearly related
    at all?" -- which is exactly the precondition for an ITD to be meaningful, and
    it is high for a clean tone and low for uncorrelated wind noise.

    Welch estimate with `nseg` half-overlapping segments. With K independent
    segments the expected coherence of two UNRELATED signals is 1/K, so that
    floor is subtracted and the result rescaled to [0, 1].
    """
    n = len(a)
    if n < 4 * nseg:
        return 0.0
    seg = int(2 * n / (nseg + 1))
    hop = seg // 2
    win = np.hanning(seg)
    nfft = 1 << int(np.ceil(np.log2(seg)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if not m.any():
        return 0.0
    Pxx = Pyy = Pxy = 0.0
    k = 0
    for i in range(0, n - seg + 1, hop):
        A = np.fft.rfft((a[i:i + seg] - a[i:i + seg].mean()) * win, nfft)
        B = np.fft.rfft((b[i:i + seg] - b[i:i + seg].mean()) * win, nfft)
        Pxx = Pxx + np.abs(A) ** 2
        Pyy = Pyy + np.abs(B) ** 2
        Pxy = Pxy + A * np.conj(B)
        k += 1
    if k < 2:
        return 0.0
    coh = np.abs(Pxy[m]) ** 2 / (Pxx[m] * Pyy[m] + 1e-30)
    # weight by signal energy: bands with no energy carry no information
    wgt = (Pxx[m] + Pyy[m])
    c = float(np.sum(coh * wgt) / (np.sum(wgt) + 1e-30))
    floor = 1.0 / k
    return float(np.clip((c - floor) / (1.0 - floor), 0.0, 1.0))


def _welch_cross(a, b, fs, f_lo, f_hi, nseg=6):
    """One Welch pass -> (freqs_in_band, coherence, mean power, n_segments)."""
    n = len(a)
    if n < 4 * nseg:
        return None
    seg = int(2 * n / (nseg + 1))
    hop = max(1, seg // 2)
    win = np.hanning(seg)
    nfft = 1 << int(np.ceil(np.log2(seg)))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    if m.sum() < 4:
        return None
    Pxx = Pyy = Pxy = 0.0
    k = 0
    for i in range(0, n - seg + 1, hop):
        A = np.fft.rfft((a[i:i + seg] - a[i:i + seg].mean()) * win, nfft)
        B = np.fft.rfft((b[i:i + seg] - b[i:i + seg].mean()) * win, nfft)
        Pxx = Pxx + np.abs(A) ** 2
        Pyy = Pyy + np.abs(B) ** 2
        Pxy = Pxy + A * np.conj(B)
        k += 1
    if k < 2:
        return None
    coh = np.abs(Pxy[m]) ** 2 / (Pxx[m] * Pyy[m] + 1e-30)
    pwr = 0.5 * (Pxx[m] + Pyy[m])
    return f[m], coh, pwr, k


def coherence_confidence(a, b, fs, f_lo=120.0, f_hi=1500.0, nseg=6):
    """Band-averaged magnitude-squared coherence, bias-corrected.

    Replaces the peak-to-sidelobe margin, which is wrong here: for a tonal source
    the GCC surface is periodic, so that margin collapsed even when the bearing
    was perfect (every arm reported 100% fallback on engine and rotor in E05).
    With K segments, two unrelated signals give E[coherence] = 1/K, so that floor
    is removed.
    """
    r = _welch_cross(a, b, fs, f_lo, f_hi, nseg)
    if r is None:
        return 0.0
    _, coh, pwr, k = r
    c = float(np.sum(coh * pwr) / (np.sum(pwr) + 1e-30))
    floor = 1.0 / k
    return float(np.clip((c - floor) / (1.0 - floor), 0.0, 1.0))


def coherent_spread(a, b, fs, f_lo=120.0, f_hi=1500.0, nseg=6):
    """Spectral flatness of the COHERENT power, coh(f) * P(f).

    A two-microphone ITD estimator is ambiguous for a line spectrum: measured, a
    pure harmonic source gives 31-40 deg bearing RMSE and more data does not help
    (31.4 deg at 16 ms, 31.6 deg at 500 ms) -- a bias, not noise. Adding 20%
    broadband drops it to 1.3 deg.

    The flatness must be measured on the COHERENT part, not on one channel.
    Independent noise raises a single channel's flatness without making the delay
    resolvable, and an earlier version of this function was fooled exactly that
    way: confidence ROSE from 0.24 to 0.84 as SNR fell, while the error stayed at
    30 deg. Coherence weighting removes the independent noise from the estimate.
    """
    r = _welch_cross(a, b, fs, f_lo, f_hi, nseg)
    if r is None:
        return 0.0
    _, coh, pwr, k = r
    c = np.clip(coh - 1.0 / k, 0.0, 1.0) * pwr + 1e-20
    return float(np.clip(np.exp(np.mean(np.log(c))) / (np.mean(c) + 1e-30), 0.0, 1.0))


def bearing_confidence(a, b, fs, f_lo=120.0, f_hi=1500.0, nseg=6,
                       spread_knee=0.15):
    """Confidence that the estimated bearing is usable.

        conf = coherence  x  f(coherent spread)

    Coherence answers "are the channels linearly related" -- high for a clean
    tone. Spread answers "is the delay resolvable" -- low for a clean tone, where
    the bearing really is wrong by ~30 deg. Only the product is a useful gate,
    and it is validated by requiring that confidence PREDICT bearing error.
    """
    coh = coherence_confidence(a, b, fs, f_lo, f_hi, nseg)
    sp = coherent_spread(a, b, fs, f_lo, f_hi, nseg)
    return float(np.clip(coh * np.clip(np.sqrt(sp / spread_knee), 0.0, 1.0), 0.0, 1.0))


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

    def __init__(self, theta0, tau_c=0.5, conf_floor=0.08):
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
