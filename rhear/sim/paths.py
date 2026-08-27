"""Virtual acoustics: primary path, secondary path, head geometry."""
import numpy as np
from scipy import signal
from .electronics import fractional_delay_fir

C_AIR = 343.0


def delay_impulse(delay_s, fs, ntaps=None):
    """Impulse response of a pure delay, with fractional-sample accuracy."""
    d = delay_s * fs
    whole, frac = int(np.floor(d)), d - int(np.floor(d))
    frac_fir = fractional_delay_fir(frac, ntaps=32)
    h = np.zeros(whole + len(frac_fir))
    h[whole:] = frac_fir
    return h


def _impulse(b, a, ntaps):
    """Impulse response, long enough that the tail is negligible."""
    return signal.lfilter(b, a, np.r_[1.0, np.zeros(ntaps - 1)])


def primary_path(fs, distance_m=0.07, cup_cutoff_hz=1200.0, order=2, ntaps=1024):
    """Reference mic -> ear. Propagation delay plus the cup's passive lowpass."""
    h_delay = delay_impulse(distance_m / C_AIR, fs)
    b, a = signal.butter(order, cup_cutoff_hz / (fs / 2), btype="low")
    h = np.convolve(h_delay, _impulse(b, a, ntaps))
    return h / np.linalg.norm(h)          # unit energy, so mu means the same thing everywhere


def secondary_path(fs, distance_m=0.02, f_lo=150.0, f_hi=5000.0, order=2, ntaps=4096):
    """Driver -> ear. Acoustic delay plus the driver's bandpass response.

    f_lo is the driver's low-frequency roll-off. It must not be so low that the
    impulse response fails to settle inside ntaps -- an under-truncated IIR tail
    silently corrupts every result downstream (this bit us on the first run).
    """
    h_delay = delay_impulse(distance_m / C_AIR, fs)
    b, a = signal.butter(order, [f_lo / (fs / 2), f_hi / (fs / 2)], btype="band")
    h = np.convolve(h_delay, _impulse(b, a, ntaps))
    tail = np.sum(h[-ntaps // 8:] ** 2) / np.sum(h ** 2)
    if tail > 1e-4:
        raise ValueError(f"secondary path truncated too early (tail energy {tail:.2e})")
    return h / np.linalg.norm(h)


def add_electrical_delay(s_path, delay_s, fs):
    """Total secondary path = acoustic secondary path * whole electrical chain."""
    if delay_s <= 0:
        return s_path.copy()
    return np.convolve(s_path, delay_impulse(delay_s, fs))


def primary_delay_vs_azimuth(theta_deg, mic_offset_m=0.07, head_radius_m=0.0875):
    """Reference-mic-to-ear travel time as a function of source azimuth.

    theta = 0 is straight ahead (mic faces the source: longest primary delay).
    theta = 180 is from behind (the wavefront can reach the ear first).
    Woodworth spherical-head model for the ear; a simple projection for the
    shell-mounted reference mic.
    """
    th = np.radians(theta_deg)
    a = head_radius_m
    # Woodworth: extra path to the far ear
    ear_path = a * (np.abs(th) + np.sin(np.abs(th))) if np.abs(th) <= np.pi / 2 \
        else a * (np.abs(th) + np.sin(np.pi - np.abs(th)))
    mic_path = -mic_offset_m * np.cos(th)      # mic is forward of the ear
    return (ear_path + mic_path) / C_AIR
