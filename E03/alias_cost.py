#!/usr/bin/env python3
"""What does skipping the anti-alias filter actually cost the coherence result?

The filter is the one thing on the build list that has to be bought, so the
question "can we just not fit it" deserves a number rather than a warning.

Method: build a synthetic two-microphone field at 96 kHz where the true
coherence is KNOWN by construction -- a shared source through a shell low-pass,
plus independent leakage at each mic that grows with frequency (shorter
wavelengths decorrelate across the 7 cm spacing). Then sample it three ways:

    ideal   properly decimated to 8 kHz  -> the truth
    RC      one pole at 3.4 kHz, then decimated  -> what we would build
    none    decimated with no filter at all  -> what skipping it gives

and compare the cancellation ceiling each one reports.

Run for two noise sources, because the answer depends on what is playing:
white noise has as much energy above 4 kHz as below, pink noise has much less.

    /opt/anaconda3/bin/python E03/alias_cost.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, lfilter, coherence, resample_poly

FS_HI, FS = 96000, 8000
SECONDS = 30
BANDS = [(100, 500), (500, 1000), (1000, 2000), (2000, 4000)]
rng = np.random.default_rng(3)


def pink(n):
    """1/f noise -- closer to a fan, an engine, or road noise than white is."""
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS_HI)
    f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n)


def mic_chain(x):
    """What the microphone can actually deliver.

    An electret capsule plus a MAX4466 is not flat to 48 kHz -- the capsule
    rolls off in the high teens. Leaving the synthetic source flat to Nyquist
    at 96 kHz would invent aliasing energy that no real microphone produces,
    and would exaggerate the cost of skipping the filter. Band-limit first,
    then compare."""
    b, a = butter(3, 18000 / (FS_HI / 2))
    return lfilter(b, a, x)


def field(source):
    """Two mics, 7 cm apart through the cup shell.

    The shared part reaches the inner mic through the shell (a low-pass). The
    independent part grows with frequency, which is what actually destroys
    coherence on a real headset: at 4 kHz the wavelength is 8.6 cm, comparable
    to the spacing, so the two mics stop seeing the same thing."""
    b, a = butter(2, 1500 / (FS_HI / 2))
    shared = lfilter(b, a, source)
    shared /= shared.std()
    bh, ah = butter(2, 800 / (FS_HI / 2), btype="high")
    leak = lfilter(bh, ah, rng.standard_normal(len(source)))
    leak /= leak.std()
    ref = source / source.std()
    err = shared + 0.55 * leak
    return mic_chain(ref), mic_chain(err)


def rc(x, fc=3386.0):
    """One pole, exactly what 1 kOhm into 47 nF does."""
    b, a = butter(1, fc / (FS_HI / 2))
    return lfilter(b, a, x)


def to_fs(x, filtered):
    if filtered:                      # proper decimation: the honest reference
        return resample_poly(x, 1, FS_HI // FS)
    return x[::FS_HI // FS]           # plain decimation: everything folds


def bands_of(ref, err):
    f, cxy = coherence(ref, err, fs=FS, nperseg=1024)
    out = []
    for lo, hi in BANDS:
        s = (f >= lo) & (f < hi)
        c = float(np.mean(cxy[s]))
        out.append(-10 * np.log10(max(1 - c, 1e-6)))
    return out


n = FS_HI * SECONDS
print(f"\n{SECONDS}s, 7 cm spacing, coherence over {len(BANDS)} bands\n")

for name, src in (("WHITE noise  (radio static, hiss)", rng.standard_normal(n)),
                  ("PINK noise   (fan, engine, road)", pink(n))):
    ref_hi, err_hi = field(src)
    truth = bands_of(to_fs(ref_hi, True), to_fs(err_hi, True))
    with_rc = bands_of(to_fs(rc(ref_hi), False), to_fs(rc(err_hi), False))
    none = bands_of(to_fs(ref_hi, False), to_fs(err_hi, False))

    print(name)
    print(f"  {'band':>14s}{'truth':>9s}{'with RC':>10s}{'no filter':>11s}"
          f"{'error':>9s}")
    for i, (lo, hi) in enumerate(BANDS):
        err_db = none[i] - truth[i]
        print(f"  {f'{lo}-{hi} Hz':>14s}{truth[i]:8.1f}{with_rc[i]:10.1f}"
              f"{none[i]:11.1f}{err_db:+9.1f} dB")
    worst = max(abs(none[i] - truth[i]) for i in range(len(BANDS)))
    worst_rc = max(abs(with_rc[i] - truth[i]) for i in range(len(BANDS)))
    print(f"  worst error without the filter : {worst:.1f} dB")
    print(f"  worst error with it            : {worst_rc:.1f} dB\n")
