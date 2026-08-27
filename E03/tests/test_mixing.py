"""Code-path tests for the mixture engine.

IMPORTANT: these use synthetic stand-in audio to exercise the CODE. They are not
a dataset and produce no dataset claim. The real dataset is built from real
recordings by the Colab notebook, and its properties are reported by
rhear_data/report.py from the manifest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rhear_data import mixing, headset, splits

FS = 16_000


def _stub(n, kind, rng):
    """Stand-in audio. NOT a real recording -- for exercising code only."""
    t = np.arange(n) / FS
    if kind == "speech":
        return (np.sin(2 * np.pi * 130 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
                + 0.2 * rng.standard_normal(n))
    return rng.standard_normal(n)


def _mk(rng, dur=4.0, **kw):
    n = int(dur * FS) + 8000
    sp = dict(audio=_stub(n, "speech", rng), speaker="S1", utt="S1-0001")
    nz = [dict(audio=_stub(n, "noise", rng), cls="vehicle", source="stub:v1"),
          dict(audio=_stub(n, "noise", rng), cls="gunshot", source="stub:g1")]
    return mixing.make_mixture(sp, nz, FS, rng, dur_s=dur, **kw)


def test_shapes_and_pairing():
    rng = np.random.default_rng(0)
    x, y, m = _mk(rng)
    assert x.shape == y.shape == (int(4.0 * FS),)
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert np.max(np.abs(x)) <= 1.0 and np.max(np.abs(y)) <= 1.0
    assert m["fs"] == FS and m["provenance"] == "synthetically mixed from real recordings"


def test_snr_is_actually_achieved():
    """The requested SNR must be measurable in the output, over the speech-active
    region, to within a tolerance set by the post-mix self-noise and clipping."""
    errs = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        x, y, m = _mk(rng, clip_prob=0.0, distort_prob=0.0)
        noise = x - y
        act = np.abs(y) > 0.02 * np.max(np.abs(y))
        if act.sum() < FS // 2:
            continue
        got = 20 * np.log10(mixing.rms(y[act]) / (mixing.rms(noise[act]) + 1e-12))
        errs.append(got - m["snr_db"])
    e = float(np.mean(np.abs(errs)))
    assert e < 3.0, f"mean |SNR error| {e:.2f} dB over {len(errs)} samples"
    return e, len(errs)


def test_target_is_noise_free():
    rng = np.random.default_rng(3)
    x, y, m = _mk(rng)
    assert not np.allclose(x, y), "mixture and target are identical"
    assert mixing.rms(x - y) > 1e-4, "no noise was added"


def test_level_trajectories_are_not_constant():
    rng = np.random.default_rng(1)
    kinds, varying = [], 0
    for _ in range(200):
        g, k = mixing.level_trajectory(FS, FS, rng)
        kinds.append(k)
        if np.std(g) > 0.02:
            varying += 1
        assert abs(np.mean(g) - 1.0) < 1e-6, "trajectory must have unit mean"
    from collections import Counter
    c = Counter(kinds)
    assert varying / 200 > 0.6, f"only {varying/200:.0%} of trajectories vary"
    return dict(c), varying / 200


def test_speech_does_not_fill_the_clip():
    rng = np.random.default_rng(2)
    fr = [mixing.speech_activity(4 * FS, FS, rng)[1] for _ in range(100)]
    assert np.mean(fr) < 0.95, "speech fills every clip -- no speech-absent regions"
    assert np.min(fr) < 0.7, "no clip has a substantial speech-absent region"
    return float(np.mean(fr)), float(np.min(fr))


def test_clipping_and_distortion_are_recorded():
    rng = np.random.default_rng(7)
    seen_clip = seen_dist = 0
    for _ in range(120):
        _, _, m = _mk(rng, clip_prob=0.5, distort_prob=0.5)
        seen_clip += m["clipping"] is not None
        seen_dist += m["preamp_distortion_k"] is not None
        if m["clipping"]:
            assert m["clipping"]["clipped_frac"] >= 0.0
    assert seen_clip > 30 and seen_dist > 30
    return seen_clip, seen_dist


def test_stationarity_labelling():
    assert mixing.classify_stationarity("gunshot") == "impulsive"
    assert mixing.classify_stationarity("vehicle") == "stationary"
    assert mixing.classify_stationarity("helicopter") == "nonstationary"
    rng = np.random.default_rng(0)
    _, _, m = _mk(rng)
    assert m["mixture_stationarity"] == "mixed"     # vehicle + gunshot


def test_rir_preserves_length_and_alignment():
    rng = np.random.default_rng(4)
    x = _stub(FS, "speech", rng)
    rir = np.zeros(400); rir[37] = 1.0; rir[80] = 0.4
    y = headset.apply_rir(x, rir)
    assert len(y) == len(x)
    lag = int(np.argmax(np.correlate(y[:2000], x[:2000], "full"))) - 1999
    assert abs(lag) <= 2, f"direct path shifted by {lag} samples"


def test_split_groups_are_disjoint():
    rng = np.random.default_rng(0)
    g = [f"src{i}" for i in range(200)]
    s = splits.split_groups(g, rng)
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"]) \
        and not (s["val"] & s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 200


if __name__ == "__main__":
    test_shapes_and_pairing();     print("  shapes / pairing .............. PASS")
    e, n = test_snr_is_actually_achieved()
    print(f"  requested SNR is achieved ..... PASS (mean |error| {e:.2f} dB over {n})")
    test_target_is_noise_free();   print("  target is noise-free .......... PASS")
    c, v = test_level_trajectories_are_not_constant()
    print(f"  level trajectories vary ....... PASS ({v:.0%} non-constant, {c})")
    a, mn = test_speech_does_not_fill_the_clip()
    print(f"  speech/noise timing overlap ... PASS (mean active {a:.0%}, min {mn:.0%})")
    sc, sd = test_clipping_and_distortion_are_recorded()
    print(f"  clipping/distortion recorded .. PASS ({sc} clipped, {sd} distorted of 120)")
    test_stationarity_labelling(); print("  stationarity labelling ........ PASS")
    test_rir_preserves_length_and_alignment()
    print("  RIR keeps length + alignment .. PASS")
    test_split_groups_are_disjoint(); print("  split groups disjoint ......... PASS")
    print("\n  NOTE: these exercise CODE using synthetic stand-in audio.")
    print("  They make no claim about the dataset, which is built from real")
    print("  recordings in the Colab notebook.")
