#!/usr/bin/env python3
"""Pick the anti-alias design by measurement, not by picking a round number.

alias_cost.py showed the first choice -- 1 kOhm into 47 nF, corner 3.4 kHz,
sampling at 8 kHz -- still leaves tens of dB of error in the coherence result.
One pole at 3.4 kHz is only about 5 dB down at 6 kHz, and 6 kHz folds straight
onto 2 kHz. The filter was too gentle and the sample rate too low.

The insight that makes this cheap: COHERENCE IS NORMALISED. Both channels get
the same R and the same C, so the filter's response divides out of gamma^2
exactly. Rolling off hard inside the measurement band costs nothing except
signal-to-noise against the ADC's own floor -- which means the corner can go far
lower than "flat to 4 kHz" instinct suggests.

So this sweeps corner frequency and sample rate together and reports the error
each combination makes against a known ground truth.

    /opt/anaconda3/bin/python E03/alias_design.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, lfilter, coherence, resample_poly

FS_HI = 96000
SECONDS = 30
BANDS = [(100, 500), (500, 1000), (1000, 2000), (2000, 4000)]
rng = np.random.default_rng(3)


def pink(n):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS_HI); f[0] = f[1]
    return np.fft.irfft(X / np.sqrt(f), n)


def mic_chain(x):
    b, a = butter(3, 18000 / (FS_HI / 2))
    return lfilter(b, a, x)


def field(source):
    b, a = butter(2, 1500 / (FS_HI / 2))
    shared = lfilter(b, a, source); shared /= shared.std()
    bh, ah = butter(2, 800 / (FS_HI / 2), btype="high")
    leak = lfilter(bh, ah, rng.standard_normal(len(source))); leak /= leak.std()
    return mic_chain(source / source.std()), mic_chain(shared + 0.55 * leak)


def rc(x, fc, poles):
    """One RC section per pole. fc >= Nyquist means no filter at all."""
    if fc >= FS_HI / 2:
        return x
    for _ in range(poles):
        b, a = butter(1, fc / (FS_HI / 2))
        x = lfilter(b, a, x)
    return x


def bands_of(ref, err, fs):
    f, cxy = coherence(ref, err, fs=fs, nperseg=1024)
    out = []
    for lo, hi in BANDS:
        s = (f >= lo) & (f < hi)
        c = float(np.mean(cxy[s]))
        out.append(-10 * np.log10(max(1 - c, 1e-6)))
    return out


n = FS_HI * SECONDS
src = pink(n)                      # the realistic case; white is the worst case
ref_hi, err_hi = field(src)
truth = bands_of(resample_poly(ref_hi, 1, 12), resample_poly(err_hi, 1, 12), 8000)

print(f"\nground truth, pink noise, {SECONDS}s")
for (lo, hi), t in zip(BANDS, truth):
    print(f"  {lo:5d}-{hi:5d} Hz   {t:5.1f} dB")

# ONLY parts that are actually in the box: 1 kOhm resistors, 100 nF (the
# "0.1 uF" from bill 2921), 10 nF (the biggest disc in the assorted pF box),
# and 10 uF electrolytics. No 47 nF, no 150 nF -- the assorted box tops out at
# 10 nF. Two 1 kOhm in series make 2 kOhm, which is free.
def fc(r, c):
    return 1.0 / (2 * np.pi * r * c)

CAND = [
    ("nothing at all",        1e9,               1,  8000, "skip it entirely"),
    ("nothing at all",        1e9,               1, 16000, "skip it, sample faster"),
    ("1 kΩ + 10 nF",          fc(1e3, 10e-9),    1,  8000, "biggest cap in the pF box"),
    ("1 kΩ + 100 nF",         fc(1e3, 100e-9),   1,  8000, "the 0.1 µF from bill 2921"),
    ("1 kΩ + 100 nF",         fc(1e3, 100e-9),   1, 16000, "same, sample faster"),
    ("2 kΩ + 100 nF",         fc(2e3, 100e-9),   1,  8000, "two 1 kΩ in series"),
    ("2 kΩ + 100 nF",         fc(2e3, 100e-9),   1, 16000, "series R, faster"),
    ("2x (1 kΩ + 100 nF)",    fc(1e3, 100e-9),   2,  8000, "two RC sections"),
    ("2x (2 kΩ + 100 nF)",    fc(2e3, 100e-9),   2,  8000, "two sections, series R"),
    ("1 kΩ + 10 µF",          fc(1e3, 10e-6),    1,  8000, "the electrolytic — far too low?"),
]

print(f"\n{'filter':<22s}{'corner':>8s}{'poles':>7s}{'rate':>8s}"
      f"{'worst err':>11s}   verdict")
print("-" * 78)
best = None
for name, fcut, poles, fs, note in CAND:
    fc = fcut
    dec = FS_HI // fs
    r = rc(ref_hi, fc, poles)[::dec]
    e = rc(err_hi, fc, poles)[::dec]
    got = bands_of(r, e, fs)
    worst = max(abs(g - t) for g, t in zip(got, truth))
    ok = "USABLE" if worst <= 1.5 else ("marginal" if worst <= 3.0 else "no")
    if best is None or worst < best[0]:
        best = (worst, name, fc, poles, fs, note)
    print(f"{name:<22s}{fc:8.0f}{poles:7d}{fs:8d}{worst:10.1f} dB   {ok}  — {note}")

w, name, fc, poles, fs, note = best
print(f"\nbest: {name}, {poles} pole(s) at {fc:.0f} Hz, sampled at {fs} Hz "
      f"— worst band error {w:.1f} dB")
print("\nCoherence is normalised, so an identical filter on both channels "
      "divides out\nof gamma^2 exactly. Rolling off hard inside the band costs "
      "nothing but ADC\nsignal-to-noise, which is why the low corners win.")
