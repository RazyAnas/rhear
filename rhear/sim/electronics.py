"""Virtual electronics.

Every latency number this module reports is *derived* from a filter design, not
asserted. That is the point: we want the simulator to independently reproduce the
500-600 us figure quoted for generic audio codecs, which is how we know the model
is trustworthy before we spend money.
"""
import numpy as np
from scipy import signal


def decimation_chain_group_delay(fs_modulator=3_072_000, fs_out=48_000,
                                 cic_order=4, cic_decim=16, verbose=False):
    """Group delay of a typical sigma-delta audio decimation chain.

    Stage 1: CIC of order `cic_order`, decimation `cic_decim`.
             A CIC with N stages and rate change R has group delay
             N*(R-1)/2 samples at the *input* rate.
    Stage 2/3: linear-phase halfband FIRs, each decimating by 2.
               Group delay (Ntaps-1)/2 samples at that stage's input rate.

    Returns total group delay in seconds.
    """
    remaining = (fs_modulator // cic_decim) // fs_out          # extra decimation needed
    stages = []

    gd_cic = cic_order * (cic_decim - 1) / 2.0 / fs_modulator
    stages.append(("CIC order %d, R=%d" % (cic_order, cic_decim), gd_cic))

    fs_stage = fs_modulator / cic_decim
    n_half = int(np.log2(remaining)) if remaining >= 2 else 0
    for i in range(n_half):
        ntaps = 47 if i == 0 else 95         # later stages need sharper transition
        gd = (ntaps - 1) / 2.0 / fs_stage
        stages.append(("halfband FIR %d taps @ %.0f kHz" % (ntaps, fs_stage / 1e3), gd))
        fs_stage /= 2

    total = sum(g for _, g in stages)
    if verbose:
        for name, g in stages:
            print(f"    {name:38s} {g*1e6:8.1f} us")
        print(f"    {'TOTAL':38s} {total*1e6:8.1f} us")
    return total


def fractional_delay_fir(delay_samples, ntaps=32):
    """Windowed-sinc fractional delay filter. Lets us represent sub-sample delays."""
    n = np.arange(ntaps)
    centre = (ntaps - 1) / 2.0
    h = np.sinc(n - centre - delay_samples) * signal.windows.blackman(ntaps)
    s = h.sum()
    return h / s if abs(s) > 1e-12 else h


def soft_clip_aop(x, aop_dbspl=135.0, ref_dbspl=94.0, ref_amplitude=1.0):
    """Microphone AOP saturation, modelled as a tanh knee at the AOP level.

    aop_dbspl is where THD hits 10%. ref_amplitude is the digital amplitude
    corresponding to ref_dbspl (94 dB SPL = 1 Pa, the usual sensitivity point).
    """
    ceiling = ref_amplitude * 10 ** ((aop_dbspl - ref_dbspl) / 20.0)
    return ceiling * np.tanh(x / ceiling)


def mic_self_noise(n, snr_db=73.0, ref_amplitude=1.0, rng=None):
    """Analog MEMS self noise. SNR is quoted re 94 dB SPL (1 Pa)."""
    rng = rng or np.random.default_rng(0)
    sigma = ref_amplitude * 10 ** (-snr_db / 20.0)
    return sigma * rng.standard_normal(n)


# Catalogue of the parts we actually costed, so an experiment can name a part
# instead of a number. Latencies in seconds.
CODECS = {
    "ADAU1777/1787 (ANC codec)":      5e-6,
    "generic 48 kHz codec":           None,   # derived by decimation_chain_group_delay()
    "USB audio interface (laptop)":   6e-3,
    "analog ANC IC (AS3415 class)":   0.0,
}
