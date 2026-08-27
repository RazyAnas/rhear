"""Headset realism: the chain between the acoustic field and the ADC.

Everything here models a real effect of the RHEAR hardware defined in
docs/04-hardware-design.md. Nothing is added merely to make the task harder --
each transform has a named physical cause and a parameter range taken from the
part selection, and each one is recorded in the sample's metadata.
"""
import numpy as np
from scipy import signal


def mic_frequency_response(x, fs, rng, hp_hz=None, lp_hz=None, tilt_db=None):
    """Unit-to-unit variation in the boom microphone's response.

    Cause: electret/MEMS capsule tolerance plus the windshield and boom cavity.
    Modelled as a gentle high-pass (port/vent), a gentle low-pass (capsule mass)
    and a broad spectral tilt.
    """
    hp = hp_hz if hp_hz is not None else float(rng.uniform(40, 140))
    lp = lp_hz if lp_hz is not None else float(rng.uniform(6500, 7800))
    tilt = tilt_db if tilt_db is not None else float(rng.uniform(-3.0, 3.0))
    b, a = signal.butter(1, hp / (fs / 2), btype="high")
    y = signal.lfilter(b, a, x)
    lp = min(lp, 0.95 * fs / 2)
    b, a = signal.butter(1, lp / (fs / 2), btype="low")
    y = signal.lfilter(b, a, y)
    n = 512
    f = np.linspace(0, 1, n // 2 + 1)
    g = 10 ** ((tilt * (f - 0.5)) / 20.0)
    h = np.fft.irfft(g, n)
    h = np.roll(h, n // 2) * np.hanning(n)
    y = signal.lfilter(h / np.sum(np.abs(h)) * np.sum(np.abs(h)), [1.0], y)
    return y, dict(mic_hp_hz=hp, mic_lp_hz=lp, mic_tilt_db=tilt)


def mic_self_noise(n, rng, snr_ref_db=None):
    """Analog MEMS self-noise. IM73A135 is 73 dB SNR re 94 dB SPL."""
    s = snr_ref_db if snr_ref_db is not None else float(rng.uniform(64, 74))
    return s, (10 ** (-s / 20.0)) * rng.standard_normal(n)


def soft_clip(x, headroom_db, knee="tanh"):
    """Preamp / ADC saturation. headroom_db is how far the peak sits below the
    clipping ceiling; small values mean the sample really does clip."""
    peak = np.max(np.abs(x)) + 1e-12
    ceiling = peak * 10 ** (-headroom_db / 20.0)
    if knee == "hard":
        y = np.clip(x, -ceiling, ceiling)
    else:
        y = ceiling * np.tanh(x / ceiling)
    frac = float(np.mean(np.abs(x) > ceiling))
    return y, frac


def apply_rir(x, rir, keep_len=True):
    """Convolve with a measured room impulse response, aligned so the direct
    path is not delayed (otherwise the target and the mixture drift apart)."""
    r = np.asarray(rir, float)
    r = r / (np.max(np.abs(r)) + 1e-12)
    d = int(np.argmax(np.abs(r)))
    y = signal.fftconvolve(x, r)[d:]
    return y[:len(x)] if keep_len else y


def level_mismatch_db(rng, lo=-4.0, hi=4.0):
    """Gain error between capture sessions / units."""
    return float(rng.uniform(lo, hi))
