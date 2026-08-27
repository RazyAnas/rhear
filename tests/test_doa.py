"""Unit tests for bearing estimation (G5 requirements 3 and 4).

Requirement 4: GCC-PHAT with band limiting and fractional-delay interpolation
must recover known source angles.

Requirement 3: the confidence statistic must PREDICT bearing error. That is the
only property worth testing -- an early draft of this file asserted "confidence
must be high for a clean tonal source", which was wrong: for a clean tonal source
the bearing really is wrong by ~30 deg, so confidence there SHOULD be low.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rhear.sim.geometry import render, theta_from_itd, mic_spacing, C_AIR
from rhear.core.doa import (gcc_phat, coherence_confidence, coherent_spread,
                            bearing_confidence)
from rhear.core import signals as sg

FS = 48_000
BAND = (120.0, 1500.0)
GATE = 0.4


def _mk(kind, n, rng):
    if kind == "broadband":
        return sg.broadband(n, FS, *BAND, rng=rng)
    if kind == "tonal":
        return sg.harmonic(n, FS, f0=120.0, n_harm=6, rng=rng)
    if kind == "engine":
        return sg.harmonic(n, FS, f0=50.0, n_harm=10, rng=rng)
    if kind == "mixed":
        return (0.8 * sg.harmonic(n, FS, f0=120.0, n_harm=6, rng=rng)
                + 0.2 * sg.broadband(n, FS, *BAND, rng=rng))
    raise ValueError(kind)


def probe(kind, th, snr_db, frame=768, rng=None):
    rng = rng or np.random.default_rng(4)
    n = frame + 6000
    src = _mk(kind, n, rng)
    L, R, _ = render(src, np.full(n, float(th)), FS)
    a, b = L[3000:3000 + frame], R[3000:3000 + frame]
    if snr_db is not None:
        nz = np.std(a) * 10 ** (-snr_db / 20.0)
        a = a + nz * rng.standard_normal(frame)
        b = b + nz * rng.standard_normal(frame)
    tau, _ = gcc_phat(a, b, FS, f_lo=BAND[0], f_hi=BAND[1])
    return theta_from_itd(tau) - th, bearing_confidence(a, b, FS, *BAND)


# ---------------- requirement 4 ----------------
def test_known_angles_broadband():
    errs = [probe("broadband", t, 40)[0] for t in (-70, -45, -20, 0, 20, 45, 70)]
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    assert rmse < 2.0, f"broadband RMSE {rmse:.2f} deg"
    return rmse


def test_known_angles_mixed():
    errs = [probe("mixed", t, 40)[0] for t in (-60, -30, 0, 30, 60)]
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    assert rmse < 6.0, f"mixed RMSE {rmse:.2f} deg"
    return rmse


def test_estimates_physically_bounded():
    for t in (-80, -40, 0, 40, 80):
        e, _ = probe("broadband", t, 40)
        assert -91 <= e + t <= 91
    assert abs(mic_spacing() / C_AIR - 0.205 / 343) < 1e-4


def test_pure_line_spectrum_is_ambiguous_and_does_not_improve_with_data():
    """Documents the limitation rather than hiding it."""
    r16 = float(np.sqrt(np.mean([probe("engine", t, 40, 768)[0] ** 2
                                 for t in (-60, -30, 30, 60)])))
    r500 = float(np.sqrt(np.mean([probe("engine", t, 40, 24000)[0] ** 2
                                  for t in (-60, -30, 30, 60)])))
    assert r16 > 15 and r500 > 15, f"expected persistent bias, got {r16:.1f}/{r500:.1f}"
    assert abs(r16 - r500) < 0.6 * r16, "should be a bias, not noise"
    return r16, r500


# ---------------- requirement 3 ----------------
def test_confidence_predicts_error():
    pts = []
    for kind in ("broadband", "tonal", "engine", "mixed"):
        for snr in (40, 10, 0, -10):
            e = c = 0.0
            for t in (-60, -30, 0, 30, 60):
                de, dc = probe(kind, t, snr)
                e += abs(de) / 5; c += dc / 5
            pts.append((c, e))
    c = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
    r = float(np.corrcoef(c, e)[0, 1])
    acc, rej = e[c > GATE], e[c <= GATE]
    assert r < -0.6, f"correlation {r:+.3f} not strongly negative"
    assert acc.mean() < 8.0, f"accepted mean error {acc.mean():.1f} deg"
    assert rej.mean() > 2.5 * acc.mean(), "gate does not separate"
    return r, acc, rej


def test_independent_channels_give_zero_confidence():
    rng = np.random.default_rng(1)
    c = bearing_confidence(rng.standard_normal(768), rng.standard_normal(768),
                           FS, *BAND)
    assert c < 0.1, f"independent channels gave {c:.3f}"
    return c


def test_coherence_high_for_clean_tone_but_spread_low():
    """The two factors must behave differently -- that is why the product works."""
    rng = np.random.default_rng(4)
    n = 768 + 6000
    src = _mk("engine", n, rng)
    L, R, _ = render(src, np.full(n, 30.0), FS)
    a, b = L[3000:3768], R[3000:3768]
    coh = coherence_confidence(a, b, FS, *BAND)
    sp = coherent_spread(a, b, FS, *BAND)
    assert coh > 0.7, f"coherence {coh:.3f} should be high for a clean tone"
    assert sp < 0.05, f"coherent spread {sp:.4f} should be low for a line spectrum"
    return coh, sp


if __name__ == "__main__":
    print(f"  known angles, broadband ....... PASS (RMSE {test_known_angles_broadband():.2f} deg)")
    print(f"  known angles, harmonic+bb ..... PASS (RMSE {test_known_angles_mixed():.2f} deg)")
    test_estimates_physically_bounded(); print("  estimates physically bounded .. PASS")
    a, b = test_pure_line_spectrum_is_ambiguous_and_does_not_improve_with_data()
    print(f"  line spectrum is ambiguous .... PASS ({a:.1f} deg @16ms, {b:.1f} deg @500ms"
          f" -- a bias, not noise)")
    coh, sp = test_coherence_high_for_clean_tone_but_spread_low()
    print(f"  coherence high / spread low ... PASS (coh {coh:.2f}, spread {sp:.4f})")
    r, acc, rej = test_confidence_predicts_error()
    print(f"  confidence predicts error ..... PASS (corr {r:+.3f})")
    print(f"     gate {GATE}: accepted mean |err| {acc.mean():.1f} deg (n={len(acc)}) | "
          f"rejected {rej.mean():.1f} deg (n={len(rej)})")
    print(f"  independent channels -> ~0 .... PASS ({test_independent_channels_give_zero_confidence():.3f})")
