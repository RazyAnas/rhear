#!/usr/bin/env python3
"""E07 - is a STRUCTURAL reference better than an acoustic one?

The idea, from active structural acoustic control: sense the disturbance on
the structure (accelerometer / piezo) instead of in the air. We already carry
an IMU for head bearing, so the sensor may cost nothing.

Three claims to test, separately:

  1. LEAD TIME. Sound crosses air at 343 m/s; vibration crosses a steel hull
     at ~5000 m/s. A structural sensor sees the disturbance milliseconds
     before it arrives at the ear as sound, where our acoustic reference buys
     204 us. E01 measured wideband cancellation collapsing to 0 dB by 300 us,
     so this is the constraint the whole architecture is built around.

  2. CLEAN REFERENCE. An acoustic reference mic also picks up the wearer's
     voice, so the canceller partly fights the signal we are paid to keep.
     An accelerometer does not hear speech at all.

  3. BANDWIDTH IS THE CATCH. A typical IMU is good to a few hundred Hz, not
     to 24 kHz. Condition C band-limits the structural reference to say
     whether the free sensor is enough or a dedicated wideband accelerometer
     is needed.

No neural network, no new model: this is the existing FxNLMS with a different
reference signal.
"""
import os, sys, json
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from rhear.core.anc import fxnlms

FS = 48_000
DUR = 4.0
N = int(FS * DUR)
C_AIR = 343.0

# geometry
D_SOURCE = 1.4          # m, source to ear through air
D_REFMIC = 0.07         # m, reference mic ahead of the ear
TAU_ELEC = 38e-6        # s, ADAU1772
TAU_SPK = 0.02 / C_AIR  # s, cup speaker to ear


def rotor(rng, n=N, f0=20.0, nh=14):
    """Blade-pass fundamental plus harmonics, amplitude-modulated."""
    t = np.arange(n) / FS
    x = np.zeros(n)
    for k in range(1, nh + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    mod = 1.0 + 0.35 * np.sin(2 * np.pi * 1.7 * t + rng.uniform(0, 6.28))
    return x * mod


def hull(rng, n=N):
    """Broadband structure-borne rumble: shaped noise, non-stationary."""
    w = rng.standard_normal(n)
    b, a = signal.butter(4, [40 / (FS / 2), 1200 / (FS / 2)], btype="band")
    x = signal.lfilter(b, a, w)
    t = np.arange(n) / FS
    return x * (1.0 + 0.5 * np.sin(2 * np.pi * 0.9 * t))


def speechlike(rng, n=N):
    """Stand-in for the wearer's voice: 300-3400 Hz, syllable-rate gating."""
    w = rng.standard_normal(n)
    b, a = signal.butter(4, [300 / (FS / 2), 3400 / (FS / 2)], btype="band")
    x = signal.lfilter(b, a, w)
    t = np.arange(n) / FS
    gate = (np.sin(2 * np.pi * 4.0 * t) > -0.2).astype(float)
    gate = signal.lfilter(np.ones(400) / 400, [1.0], gate)
    return x * gate


def delay_samples(sig_in, tau_s):
    d = int(round(tau_s * FS))
    return np.concatenate([np.zeros(d), sig_in])[:len(sig_in)]


def path(rng, taps=64):
    """A short random acoustic path, unit energy."""
    h = rng.standard_normal(taps) * np.exp(-np.arange(taps) / (taps / 3.0))
    return h / np.linalg.norm(h)


def coh_mix(src, gamma2, rng):
    """Reference with coherence gamma^2 against the source.

    The uncorrelated part is a RANDOM-PHASE SURROGATE of the source, not
    white noise. Adding white noise to a band-limited source leaves in-band
    coherence far above the requested value -- so the run beats the
    -10log10(1-gamma^2) bound and the numbers are meaningless. This is the
    same mistake the V2 coherence test made; the surrogate keeps the
    magnitude spectrum identical and randomises only phase, so gamma^2 is
    flat across frequency and equals what was asked for.
    """
    S = np.fft.rfft(src)
    ph = rng.uniform(0, 2 * np.pi, len(S))
    ph[0] = 0.0
    ind = np.fft.irfft(np.abs(S) * np.exp(1j * ph), n=len(src))
    ind = ind / (np.std(ind) + 1e-12) * np.std(src)
    return np.sqrt(gamma2) * src + np.sqrt(max(0.0, 1 - gamma2)) * ind


def measured_coherence(a, b, nper=4096):
    """Power-weighted mean magnitude-squared coherence, to verify coh_mix."""
    f, cxy = signal.coherence(a, b, fs=FS, nperseg=nper)
    _, pxx = signal.welch(a, fs=FS, nperseg=nper)
    return float(np.sum(cxy * pxx) / (np.sum(pxx) + 1e-20))


def nr_db(d, e, skip=0.5):
    """Noise reduction over the converged tail."""
    i = int(skip * len(d))
    return 10 * np.log10(np.sum(d[i:] ** 2) / (np.sum(e[i:] ** 2) + 1e-20))


def run_case(src, ref, tau_ref, s_true, s_hat, tau_primary, p_h, L=512, mu=0.05):
    """Cancel `src` arriving at tau_primary using `ref` arriving at tau_ref."""
    d = signal.lfilter(p_h, [1.0], delay_samples(src, tau_primary))
    x = delay_samples(ref, tau_ref)
    r = fxnlms(x, d, s_true, s_hat, L=L, mu=mu)
    return d, r["e"]


def secondary():
    """Speaker -> ear, including electrical delay. Unit energy."""
    tau = TAU_ELEC + TAU_SPK
    n_d = int(round(tau * FS))
    h = np.zeros(n_d + 24)
    h[n_d:] = np.exp(-np.arange(24) / 8.0)
    return h / np.linalg.norm(h)


if __name__ == "__main__":
    rng = np.random.default_rng(4242)
    s_true = secondary()
    s_hat = s_true.copy()
    p_h = path(rng)

    tau_primary = D_SOURCE / C_AIR                     # 4.08 ms
    tau_ref_ac = tau_primary - D_REFMIC / C_AIR        # acoustic ref, 204 us lead
    tau_sec = TAU_ELEC + TAU_SPK

    print("  primary %.2f ms | acoustic ref lead %.0f us | secondary %.0f us"
          % (tau_primary * 1e3, (tau_primary - tau_ref_ac) * 1e6, tau_sec * 1e6))
    print("  causality margin, acoustic reference: %.0f us\n"
          % ((tau_primary - tau_ref_ac - tau_sec) * 1e6))

    out = {}
    for name, gen in (("rotor (harmonic)", rotor), ("hull (broadband)", hull)):
        src = gen(rng)
        src = src / np.std(src)
        g2 = measured_coherence(src, coh_mix(src, 0.95, rng))
        print("  === %s ===" % name)
        print("    requested coherence 0.950 -> measured %.3f, bound %.1f dB"
              % (g2, -10 * np.log10(max(1e-9, 1 - g2))))
        print("    %-42s%10s%10s" % ("reference", "NR dB", "margin us"))

        # A. acoustic reference, clean
        ref = coh_mix(src, 0.95, rng)
        d, e = run_case(src, ref, tau_ref_ac, s_true, s_hat, tau_primary, p_h)
        a_nr = nr_db(d, e)
        print("    %-42s%10.1f%10.0f"
              % ("A  acoustic mic (7 cm lead)", a_nr,
                 (tau_primary - tau_ref_ac - tau_sec) * 1e6))
        out[name] = {"A_acoustic": a_nr, "structural": {}, "bandlimited": {}}

        # B. structural reference, wideband, lead sweep.
        # ONE reference realisation for the whole sweep -- redrawing it per
        # lead value confounds the variable under test with the random draw,
        # which is what made the first run look non-monotonic.
        ref_fixed = coh_mix(src, 0.95, rng)
        for lead_ms in (0.2, 0.5, 1.0, 2.0, 4.0):
            tau_r = max(0.0, tau_primary - lead_ms * 1e-3)
            d, e = run_case(src, ref_fixed, tau_r, s_true, s_hat, tau_primary, p_h)
            v = nr_db(d, e)
            out[name]["structural"]["%.1fms" % lead_ms] = v
            print("    %-42s%10.1f%10.0f"
                  % ("B  structural, %.1f ms lead" % lead_ms, v,
                     (lead_ms * 1e-3 - tau_sec) * 1e6))

        # B2. BREAK-EVEN COHERENCE. Conditions A and B above both use
        # gamma^2 = 0.95, which silently assumes the structural sensor has no
        # coherence advantage -- so they test lead time alone. The physical
        # claim is that an accelerometer on the source path is MORE coherent
        # with the disturbance than a distant mic in a mixed field. That
        # cannot be settled in simulation, so report what it would have to be.
        print("    %-42s%10s%10s" % ("-- break-even coherence, 1 ms lead", "", ""))
        for g in (0.90, 0.95, 0.98, 0.99):
            ref_g = coh_mix(src, g, rng)
            d, e = run_case(src, ref_g, tau_primary - 1.0e-3,
                            s_true, s_hat, tau_primary, p_h)
            v = nr_db(d, e)
            out[name].setdefault("coherence", {})["%.2f" % g] = v
            print("    %-42s%10.1f%10s"
                  % ("B2 structural, gamma^2 = %.2f" % g, v,
                     "beats A" if v > a_nr else ""))

        # C. structural but band-limited (what the IMU we already have can do)
        for fc in (200, 500, 1000):
            b, a = signal.butter(4, fc / (FS / 2), btype="low")
            ref = signal.lfilter(b, a, coh_mix(src, 0.95, rng))
            tau_r = max(0.0, tau_primary - 1.0e-3)
            d, e = run_case(src, ref, tau_r, s_true, s_hat, tau_primary, p_h)
            v = nr_db(d, e)
            out[name]["bandlimited"]["%dHz" % fc] = v
            print("    %-42s%10.1f%10s"
                  % ("C  structural, 1 ms lead, <%d Hz only" % fc, v, "-"))
        print()

    # D. speech contamination, isolated
    print("  === D. does speech in the reference hurt? (hull noise) ===")
    src = hull(rng); src = src / np.std(src)
    sp = speechlike(rng); sp = sp / np.std(sp)
    for lbl, ref, tau_r in (
            ("acoustic mic, reference contains speech",
             coh_mix(src, 0.95, rng) + 0.5 * sp, tau_ref_ac),
            ("acoustic mic, hypothetical speech-free",
             coh_mix(src, 0.95, rng), tau_ref_ac),
            ("structural, 1 ms lead, cannot hear speech",
             coh_mix(src, 0.95, rng), tau_primary - 1.0e-3)):
        d, e = run_case(src, ref, tau_r, s_true, s_hat, tau_primary, p_h)
        v = nr_db(d, e)
        out.setdefault("speech_contamination", {})[lbl] = v
        print("    %-52s%8.1f dB" % (lbl, v))

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/e07_structural.json", "w"), indent=1)
    print("\n  -> results/e07_structural.json")
