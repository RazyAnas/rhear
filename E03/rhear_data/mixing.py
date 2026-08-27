"""RHEAR mixture engine: real clean speech + real noise -> noisy speech.

The clean speech is preserved as the training target. The target is defined at
the SAME point in the chain as the mixture -- after the microphone response and
the level, but before noise, reverberant noise, self-noise and clipping -- so the
model learns to remove noise, not to invert the microphone.

Reverberation: a boom microphone sits 2-3 cm from the mouth, so the direct path
dominates and speech is very nearly anechoic. Room reverberation is therefore
applied to the NOISE, which genuinely arrives through the room, and not to the
speech. This is a deliberate physical choice, recorded in the metadata.
"""
import numpy as np
from . import headset

STATIONARITY = {
    "stationary": ["ambient", "machinery", "technical", "vehicle", "engine"],
    "nonstationary": ["crowd", "helicopter", "fighter", "siren", "wind", "footsteps"],
    "impulsive": ["gunshot", "shelling"],
}


def classify_stationarity(noise_class):
    for k, v in STATIONARITY.items():
        if noise_class in v:
            return k
    return "unknown"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-20))


def _fit(x, n, rng, allow_tile=True):
    """Take n samples from x, tiling or cropping as needed."""
    if len(x) == 0:
        return np.zeros(n)
    if len(x) >= n:
        i = int(rng.integers(0, len(x) - n + 1))
        return x[i:i + n].copy()
    if not allow_tile:
        y = np.zeros(n); y[:len(x)] = x; return y
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n].copy()


def level_trajectory(n, fs, rng, kind=None):
    """A time-varying noise gain, in linear units, mean 1.

    Static-SNR training is a known reason models fail on non-stationary noise, so
    the default is a trajectory rather than a constant. Kinds:
      constant  - flat (kept as a control condition, a minority of samples)
      ramp      - a vehicle approaching or receding
      fade      - a slow swell
      burst     - noise present only over part of the clip
    """
    kind = kind or str(rng.choice(["constant", "ramp", "fade", "burst"],
                                  p=[0.25, 0.3, 0.25, 0.2]))
    t = np.arange(n) / n
    if kind == "constant":
        g = np.ones(n)
    elif kind == "ramp":
        a, b = rng.uniform(0.25, 1.0), rng.uniform(0.25, 1.0)
        g = a + (b - a) * t
    elif kind == "fade":
        g = 0.4 + 0.6 * (0.5 - 0.5 * np.cos(2 * np.pi * (t * rng.uniform(0.5, 1.5))))
    else:
        g = np.full(n, 0.15)
        s = int(rng.uniform(0.0, 0.55) * n)
        e = min(n, s + int(rng.uniform(0.25, 0.6) * n))
        g[s:e] = 1.0
        k = max(1, int(0.02 * fs))
        g = np.convolve(g, np.hanning(k) / np.sum(np.hanning(k)), mode="same")
    return g / (np.mean(g) + 1e-12), kind


def speech_activity(n, fs, rng):
    """Realistic speech/noise timing overlap: speech does not fill the clip.

    Returns a gain envelope for the speech and the active fraction. Noise runs
    over the whole clip, so the model sees speech-absent regions -- which is what
    the speech-presence head and the conditional-computation gate are for.
    """
    onset = int(rng.uniform(0.0, 0.35) * n)
    dur = int(rng.uniform(0.45, 1.0) * (n - onset))
    g = np.zeros(n)
    g[onset:onset + dur] = 1.0
    k = max(1, int(0.01 * fs))
    w = np.hanning(k) / np.sum(np.hanning(k))
    g = np.convolve(g, w, mode="same")
    return g, float(np.mean(g > 0.5))


def make_mixture(speech, noises, fs, rng, rir=None, dur_s=4.0,
                 snr_range=(-10.0, 20.0), clip_prob=0.15, distort_prob=0.10):
    """Build one training pair.

    speech : dict(audio, speaker, utt)         REAL recording
    noises : list of dict(audio, cls, source)  REAL recordings, 1-3 of them
    rir    : measured room impulse response for the noise path, or None

    Returns (noisy, target, metadata).
    """
    n = int(dur_s * fs)
    meta = {}

    # ---- speech ----
    s = _fit(np.asarray(speech["audio"], float), n, rng, allow_tile=False)
    s = s / (np.max(np.abs(s)) + 1e-9) * 0.7
    env, active = speech_activity(n, fs, rng)
    s = s * env
    meta["speech_active_frac"] = round(active, 4)

    # ---- one microphone, applied to BOTH the speech and the mixture ----
    s_mic, mic_meta = headset.mic_frequency_response(s, fs, rng)
    meta.update(mic_meta)
    lvl = headset.level_mismatch_db(rng)
    g_lvl = 10 ** (lvl / 20.0)
    s_mic = s_mic * g_lvl
    meta["level_mismatch_db"] = round(lvl, 3)

    # the TARGET is the clean speech at this point in the chain
    target = s_mic.copy()

    # ---- noise: sum 1-3 real recordings, each with its own trajectory ----
    tot = np.zeros(n)
    layers = []
    for nz in noises:
        a = _fit(np.asarray(nz["audio"], float), n, rng)
        a = a / (rms(a) + 1e-9)
        g, kind = level_trajectory(n, fs, rng)
        a = a * g
        tot += a
        layers.append(dict(cls=nz["cls"], source=nz["source"],
                           level_traj=kind,
                           stationarity=classify_stationarity(nz["cls"])))
    meta["noise_layers"] = layers
    meta["n_noise_layers"] = len(layers)
    kinds = {l["stationarity"] for l in layers}
    meta["mixture_stationarity"] = ("mixed" if len(kinds) > 1
                                    else next(iter(kinds)) if kinds else "unknown")

    # room reverberation on the NOISE path only (see module docstring)
    if rir is not None:
        tot = headset.apply_rir(tot, rir["audio"])
        meta["rir_id"] = rir["id"]
        meta["rir_kind"] = rir["kind"]           # "real measured" | "simulated"
    else:
        meta["rir_id"] = None
        meta["rir_kind"] = "none (anechoic)"

    tot, _ = headset.mic_frequency_response(tot, fs, rng,
                                            hp_hz=mic_meta["mic_hp_hz"],
                                            lp_hz=mic_meta["mic_lp_hz"],
                                            tilt_db=mic_meta["mic_tilt_db"])
    tot = tot * g_lvl

    # ---- SNR, measured over the speech-active region only ----
    snr = float(rng.uniform(*snr_range))
    act = env > 0.5
    ps = rms(target[act]) if act.any() else rms(target)
    pn = rms(tot[act]) if act.any() else rms(tot)
    tot = tot * (ps / (pn + 1e-12)) * 10 ** (-snr / 20.0)
    meta["snr_db"] = round(snr, 3)

    noisy = target + tot

    # ---- microphone self-noise ----
    snr_ref, sn = headset.mic_self_noise(n, rng)
    peak = np.max(np.abs(noisy)) + 1e-12
    noisy = noisy + sn * peak
    meta["mic_self_noise_snr_db"] = round(snr_ref, 2)

    # ---- clipping ----
    if rng.random() < clip_prob:
        hr = float(rng.uniform(0.5, 6.0))
        noisy, frac = headset.soft_clip(noisy, hr,
                                        knee="hard" if rng.random() < 0.3 else "tanh")
        meta["clipping"] = dict(headroom_db=round(hr, 2), clipped_frac=round(frac, 5))
    else:
        meta["clipping"] = None

    # ---- optional mild nonlinear distortion (preamp) ----
    if rng.random() < distort_prob:
        k = float(rng.uniform(1.2, 2.5))
        noisy = np.tanh(k * noisy) / np.tanh(k)
        meta["preamp_distortion_k"] = round(k, 3)
    else:
        meta["preamp_distortion_k"] = None

    # ---- normalise the PAIR together so the SNR is preserved ----
    p = max(np.max(np.abs(noisy)), np.max(np.abs(target))) + 1e-12
    scale = 0.95 / p
    noisy, target = noisy * scale, target * scale

    meta["fs"] = fs
    meta["dur_s"] = dur_s
    meta["speaker"] = speech["speaker"]
    meta["utt"] = speech["utt"]
    meta["provenance"] = "synthetically mixed from real recordings"
    return noisy.astype(np.float32), target.astype(np.float32), meta
