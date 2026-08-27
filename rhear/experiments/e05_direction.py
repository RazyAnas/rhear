"""E05 - Direction and head rotation.

Gate: G5 does NOT pass on bearing accuracy. It passes only on an ANC improvement
over the fixed-reference / fixed-filter baseline.

The directional effect that matters here is NOT filter shape -- azimuth changes
the primary path by only a few samples of delay, which an adaptive filter absorbs
unaided (measured, first version of this experiment: oracle no better than
baseline). It is WHICH REFERENCE MICROPHONE IS CAUSAL AT ALL:

    left  cup reference: causal over  -84 deg .. +50 deg
    right cup reference: causal over  -50 deg .. +84 deg
    beyond +-84 deg    : neither -- the controller must disengage

That is a binary usable/unusable choice, and it is what direction buys.

Arms (identical FxNLMS refinement; only the selection differs):
  fixed         left reference, omni filter, always engaged -- the baseline
  audio only    GCC-PHAT bearing per 16 ms frame
  gyro DR       one audio fix at t=0 then dead-reckoning (drifts: that is the point)
  fusion        complementary filter, audio correction weighted by confidence
  oracle        true bearing -- the ceiling

Ablation is 2-D (rotation rate x reference SNR): rate alone cannot test fusion,
because at high SNR audio DoA is excellent and the gyro adds nothing.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import signal
from rhear.sim import paths
from rhear.sim.geometry import (render, theta_from_itd, causality_margin,
                                causal_cone, C_AIR)
from rhear.core import signals as sg
from rhear.core.doa import gcc_phat, Gyro, BearingTracker
from rhear.core.l2runtime import run_controller_multiref, train_filter, run_controller

FS = 48_000
FRAME = 768
L = 1024
ELEC_DELAY = 38e-6
MU = 0.02
SECTORS = np.array([-70.0, -40.0, -15.0, 15.0, 40.0, 70.0])
SWEEP = 78.0
DUR = 3.0
BAND = (120.0, 1200.0)            # ANC is a low-frequency technology
TAU_S = 0.02 / C_AIR


def source(n, rng):
    return (0.6 * sg.harmonic(n, FS, f0=90.0, n_harm=5, rng=rng)
            + 1.0 * sg.broadband(n, FS, *BAND, rng=rng))


def trajectory(n, rate_dps):
    t = np.arange(n) / FS
    tri = signal.sawtooth(2 * np.pi * t / (4 * SWEEP / rate_dps), width=0.5)
    th = SWEEP * tri
    return th, np.gradient(th, 1.0 / FS)


def best_ref(theta):
    """Which cup's reference is causal here? -1 = neither."""
    mL = causality_margin(theta, TAU_S, "L")
    mR = causality_margin(theta, TAU_S, "R")
    if max(mL, mR) <= 0:
        return -1
    return 0 if mL >= mR else 1


def build_case(rate, snr, rng):
    n = int(FS * DUR)
    src = source(n, rng)
    th, rate_traj = trajectory(n, rate)
    xL, xR, xE = render(src, th, FS)
    nz = np.std(xL) * 10 ** (-snr / 20.0)
    xL = xL + nz * rng.standard_normal(n)
    xR = xR + nz * rng.standard_normal(n)
    b, a = signal.butter(2, 1200.0 / (FS / 2), btype="low")
    return np.stack([xL, xR]), signal.lfilter(b, a, xE), th, rate_traj


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(9)
    s_ac = paths.secondary_path(FS, ntaps=1024)
    s = paths.add_electrical_delay(s_ac, ELEC_DELAY, FS)
    b_lp, a_lp = signal.butter(2, 1200.0 / (FS / 2), btype="low")

    print(f"causal cone, left  reference: {causal_cone(TAU_S,'L')[0]:+.0f} .. "
          f"{causal_cone(TAU_S,'L')[1]:+.0f} deg")
    print(f"causal cone, right reference: {causal_cone(TAU_S,'R')[0]:+.0f} .. "
          f"{causal_cone(TAU_S,'R')[1]:+.0f} deg")
    print(f"sweep +-{SWEEP:.0f} deg crosses both boundaries\n")

    print("Filter bank (each sector trained on its causal reference):")
    bank, bank_ref = [], []
    for sec in SECTORS:
        r = best_ref(sec)
        n = int(FS * 1.2)
        src = source(n, rng)
        xs = render(src, np.full(n, sec), FS)
        d = signal.lfilter(b_lp, a_lp, xs[2])
        w = train_filter(xs[r], d, s, L=L, mu=MU, passes=5)
        rr = run_controller(xs[r], d, s, s, L=L, w_init=w, refine=False)
        v = 10 * np.log10(np.nansum(rr["e"][FS//2:]**2) / np.sum(d[FS//2:]**2))
        print(f"    {sec:+6.0f} deg  ref={'LR'[r]}  ->{v:7.2f} dB")
        bank.append(w); bank_ref.append(r)
    n = int(FS * 1.2)
    src = source(n, rng)
    xs = render(src, np.linspace(-SWEEP, SWEEP, n), FS)
    w_omni = train_filter(xs[0], signal.lfilter(b_lp, a_lp, xs[2]), s, L=L, mu=MU, passes=5)
    print("    omni       ref=L  -> baseline, trained across the sweep\n")

    print(f"{'rate':>7}{'SNR':>7}{'arm':>10}{'bearing':>10}{'atten':>10}"
          f"{'vs base':>10}{'fallback':>10}{'disengaged':>12}")
    print("-" * 76)
    out = []
    for rate in (30.0, 100.0, 250.0):
        for snr in (20.0, 0.0):
            xs2, d, th, rt = build_case(rate, snr, rng)
            base = run_controller_multiref(
                xs2, d, s, [s, s], L=L, mu=MU, w_init=w_omni, frame=FRAME,
                selector=lambda fi, i: (None, 0, True))
            vb = 10*np.log10(np.nansum(base["e"]**2)/np.sum(d**2))
            print(f"{rate:5.0f}/s{snr:6.0f}dB{'fixed':>10}{'--':>10}{vb:7.2f} dB"
                  f"{'--':>10}{'--':>10}{'0 %':>12}")
            for mode in ("audio", "imu", "fusion", "oracle"):
                gyro = Gyro(rng=np.random.default_rng(3))
                tr = BearingTracker(theta0=th[0])
                log = {"err": [], "dis": 0, "n": 0}

                def sel(fi, i, mode=mode, tr=tr, gyro=gyro, log=log):
                    i = min(i, len(th) - 1)
                    if mode == "oracle":
                        est = th[i]
                    else:
                        a_, b_ = xs2[0, max(0, i-FRAME):i], xs2[1, max(0, i-FRAME):i]
                        if len(a_) < 64:
                            return None, None, True
                        tau, conf = gcc_phat(a_, b_, FS, f_lo=BAND[0], f_hi=BAND[1])
                        est = tr.step(FRAME/FS, gyro_dps=gyro.read(rt[i]),
                                      theta_audio=theta_from_itd(tau), conf=conf, mode=mode)
                    log["err"].append(est - th[i]); log["n"] += 1
                    r = best_ref(est)
                    if r < 0:
                        log["dis"] += 1
                        return None, None, False
                    k = int(np.argmin(np.abs(SECTORS - est)))
                    return bank[k], r, True

                r = run_controller_multiref(xs2, d, s, [s, s], L=L, mu=MU,
                                            w_init=w_omni, frame=FRAME, selector=sel)
                v = 10*np.log10(np.nansum(r["e"]**2)/np.sum(d**2))
                rmse = np.sqrt(np.mean(np.square(log["err"])))
                fb = 100.0*tr.fallbacks/max(tr.n, 1)
                dis = 100.0*log["dis"]/max(log["n"], 1)
                out.append((rate, snr, mode, rmse, v, v-vb, fb, dis))
                print(f"{'':>7}{'':>7}{mode:>10}{rmse:7.1f} d{v:8.2f} dB"
                      f"{v-vb:+8.2f} dB{fb:8.0f} %{dis:10.0f} %")
            print()

    # computational cost
    nfft = 1 << int(np.ceil(np.log2(2*FRAME)))
    gcc_ops = 3 * 5 * nfft * np.log2(nfft)
    print(f"Computational cost at {FS/FRAME:.1f} Hz frame rate:")
    print(f"    GCC-PHAT bearing : {gcc_ops*FS/FRAME/1e6:6.1f} MMAC/s "
          f"(3 x {nfft}-pt FFT per frame)")
    print(f"    gyro integration : {3*FS/FRAME/1e6:6.4f} MMAC/s  (one multiply-add)")
    print(f"    -> audio DoA is ~{gcc_ops/3:.0f}x the cost of the IMU path")
    np.savez("results/e05_direction.npz", rows=np.array(out, dtype=object))
    print(f"\n({time.time()-t0:.0f} s)  saved -> results/e05_direction.npz")
