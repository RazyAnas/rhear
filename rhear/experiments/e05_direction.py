"""E05 - Direction and head rotation, as a TRUE TWO-CUP HEADSET.

The previous version of this experiment was structurally wrong: it rendered one
ear and scored the other cup's reference against it. No ANC number from it is
quoted anywhere, here or in the docs.

This version:
  * independent left and right ear disturbances d_L, d_R
  * four feedforward pairings (refL/refR -> earL/earR), each with its own
    analytically verified causality margin (tests/test_geometry.py)
  * an independent L0 controller and filter state per ear
  * left and right attenuation scored separately and jointly

The hypothesis under test (requirement 8):

  "The primary value of direction in this headset is causal reference selection,
   not directional filter-shape optimisation."

tested as a 2x2 factorial with ORACLE bearing, so the mechanism is isolated from
the quality of the bearing estimate:

        A  fixed ipsilateral reference + omni filter        (baseline)
        B  reference SELECTION        + omni filter         (causal selection only)
        C  fixed ipsilateral reference + per-sector filter  (filter shape only)
        D  both

Then a bearing-source ablation (fixed / audio-only / IMU-only / fusion) on the
mechanism that actually won.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import signal
from rhear.sim import paths
from rhear.sim.geometry import (tau_point, margin, best_ref_for_ear, coverage,
                                theta_from_itd, C_AIR)
from rhear.core import signals as sg
from rhear.core.doa import gcc_phat, bearing_confidence, Gyro, BearingTracker
from rhear.core.l0 import StreamingANC
from rhear.core.l2runtime import train_filter

FS = 48_000
FRAME = 768                       # 16 ms -> 62.5 Hz L2 rate
L = 1024
MU = 0.005
ELEC_DELAY = 38e-6                # ADAU1772
TAU_S = 0.02 / C_AIR
BAND = (120.0, 1500.0)
# The sweep must cross the causal boundary of the ipsilateral reference, or the
# experiment cannot see the mechanism it is testing. A first run used +-78 deg,
# inside which the ipsilateral mic is causal almost everywhere, and measured only
# -0.47 dB for reference selection. A static sweep then showed the true effect:
# -9.4 dB at +60 deg and -9.5 dB at +90 deg, where ipsi is non-causal and the
# contralateral mic has ~495 us of margin.
SWEEP = 110.0
DUR = 3.0
GATE = 0.4                        # confidence gate, from tests/test_doa.py
SECTORS = np.arange(-60.0, 151.0, 30.0)     # left-ear frame; right ear mirrors
REFS = ["L", "R"]


# ---------------------------------------------------------------- plant
def source(n, rng):
    """Vehicle-like: harmonics plus broadband. The broadband part is what makes
    the bearing observable at all -- a pure line spectrum is ambiguous for a
    two-microphone ITD estimator (tests/test_doa.py)."""
    return (0.8 * sg.harmonic(n, FS, f0=90.0, n_harm=5, rng=rng)
            + 0.6 * sg.broadband(n, FS, *BAND, rng=rng))


def render_two_ear(src, theta_deg, cup_ba, snr_db=None, rng=None):
    """Both reference mics and BOTH ear signals, with time-varying delays."""
    n = len(src)
    out = {}
    base = 0.003
    idx = np.arange(n)
    for name in ("refL", "refR", "earL", "earR"):
        tau = tau_point(name, theta_deg)
        pos = idx - (np.asarray(tau) + base) * FS
        i0 = np.floor(pos).astype(int)
        fr = pos - i0
        i0c = np.clip(i0, 0, n - 2)
        y = (1 - fr) * src[i0c] + fr * src[i0c + 1]
        y[(i0 < 0) | (i0 >= n - 1)] = 0.0
        out[name] = y
    if snr_db is not None:
        rng = rng or np.random.default_rng(0)
        nz = np.std(out["refL"]) * 10 ** (-snr_db / 20.0)
        out["refL"] = out["refL"] + nz * rng.standard_normal(n)
        out["refR"] = out["refR"] + nz * rng.standard_normal(n)
    b, a = cup_ba
    dL = signal.lfilter(b, a, out["earL"])
    dR = signal.lfilter(b, a, out["earR"])
    return np.stack([out["refL"], out["refR"]]), dL, dR


def trajectory(n, rate_dps):
    t = np.arange(n) / FS
    tri = signal.sawtooth(2 * np.pi * t / (4 * SWEEP / rate_dps), width=0.5)
    th = SWEEP * tri
    return th, np.gradient(th, 1.0 / FS)


# ---------------------------------------------------------------- bank
def build_bank(s_plant, cup_ba, rng):
    """Filters for the LEFT ear at each (reference, sector) that is causal.

    The right ear is the exact mirror -- margin(R,R,+t) == margin(L,L,-t) and
    margin(L,R,+t) == margin(R,L,-t), both verified in tests/test_geometry.py --
    so one bank serves both cups with a sign flip on the bearing.
    """
    bank = {}
    print("  filter bank (left ear; right ear mirrors):")
    print(f"    {'sector':>8}{'ref':>5}{'margin us':>11}{'attenuation':>13}")
    for sec in SECTORS:
        for r in REFS:
            m = float(margin(r, "L", sec, TAU_S))
            if m <= 0:
                continue
            n = int(FS * 1.6)
            src = source(n, rng)
            xs, dL, _ = render_two_ear(src, np.full(n, sec), cup_ba, 40.0, rng)
            ri = REFS.index(r)
            w = train_filter(xs[ri], dL, s_plant, L=L, mu=MU, passes=3)
            anc = StreamingANC(L, MU, s_plant, s_plant, FS)
            anc.w = w.copy(); anc.adapting = False
            e, _ = anc.process_block(xs[ri], dL)
            k = len(e) // 2
            v = 10 * np.log10(np.sum(e[k:] ** 2) / np.sum(dL[k:] ** 2))
            bank[(r, float(sec))] = w
            print(f"    {sec:+7.0f}{r:>5}{m*1e6:11.1f}{v:11.2f} dB")
    # omnidirectional baseline filter, trained across the sweep on the ipsi ref
    n = int(FS * 1.6)
    src = source(n, rng)
    xs, dL, _ = render_two_ear(src, np.linspace(-SWEEP, SWEEP, n), cup_ba, 40.0, rng)
    bank["omni"] = train_filter(xs[0], dL, s_plant, L=L, mu=MU, passes=3)
    print("    omni      L        --   trained across the sweep (baseline)\n")
    return bank


def pick_filter(bank, ear, ref, bearing):
    """Bank lookup with the right-ear mirror applied."""
    b = bearing if ear == "L" else -bearing
    r = ref if ear == "L" else ("R" if ref == "L" else "L")
    keys = [k for k in bank if k != "omni" and k[0] == r]
    if not keys:
        return bank["omni"]
    sec = min(keys, key=lambda k: abs(k[1] - b))
    return bank[sec] if abs(sec[1] - b) <= 45 else bank["omni"]


# ---------------------------------------------------------------- estimators
class BearingSource:
    """fixed | oracle | audio | imu | fusion.

    'imu' is a rate gyro dead-reckoning from one initial audio fix -- a gyro has
    no absolute reference, so it must drift. That is the point of including it.
    """
    def __init__(self, mode, th0, rng):
        self.mode = mode
        self.gyro = Gyro(rng=rng)
        self.tr = BearingTracker(theta0=th0, conf_floor=GATE)
        self.err = []
        self.conf = []
        self.cost_s = 0.0
        self.calls = 0

    def __call__(self, xs, i, th_true, rate_true):
        if self.mode == "fixed":
            return 0.0
        if self.mode == "oracle":
            self.err.append(0.0); self.conf.append(1.0)
            return th_true
        t0 = time.perf_counter()
        a, b = xs[0, max(0, i - FRAME):i], xs[1, max(0, i - FRAME):i]
        if len(a) < 64:
            return self.tr.theta
        if self.mode == "imu":
            est = self.tr.step(FRAME / FS, gyro_dps=self.gyro.read(rate_true),
                               mode="imu")
            c = 1.0
        else:
            tau, _ = gcc_phat(a, b, FS, f_lo=BAND[0], f_hi=BAND[1])
            c = bearing_confidence(a, b, FS, *BAND)
            est = self.tr.step(FRAME / FS,
                               gyro_dps=self.gyro.read(rate_true),
                               theta_audio=theta_from_itd(tau), conf=c,
                               mode="audio" if self.mode == "audio" else "fusion")
        self.cost_s += time.perf_counter() - t0
        self.calls += 1
        self.err.append(est - th_true); self.conf.append(c)
        return est


# ---------------------------------------------------------------- runner
def run(cfg, rate_dps, snr_db, bank, s_plant, cup_ba, rng, seed=0):
    """cfg = (ref_select, filt_select, bearing_mode)."""
    ref_sel, filt_sel, bmode = cfg
    n = int(FS * DUR)
    src = source(n, rng)
    th, rate = trajectory(n, rate_dps)
    xs, dL, dR = render_two_ear(src, th, cup_ba, snr_db, rng)

    anc = {e: StreamingANC(L, MU, s_plant, s_plant, FS) for e in ("L", "R")}
    for e in anc:
        anc[e].w = bank["omni"].copy()
    est_src = BearingSource(bmode, th[0], np.random.default_rng(3 + seed))

    eL = np.empty(n); eR = np.empty(n)
    cur = {"L": ("L", None), "R": ("R", None)}
    ref_ok = {"L": 0, "R": 0}
    ref_hit = {"L": 0, "R": 0}
    nfr = 0
    engaged_frac = {"L": 0, "R": 0}
    est_track = np.zeros(n // FRAME + 1)

    for k, i in enumerate(range(0, n - FRAME + 1, FRAME)):
        est = est_src(xs, i, th[i], rate[i])
        est_track[k] = est
        nfr += 1
        for ear in ("L", "R"):
            # ---- reference choice ----
            if ref_sel:
                r, m = best_ref_for_ear(ear, est, TAU_S)
                if r is None:
                    # Nothing is causal. Do NOT mute: measured, a controller
                    # outside its causal region still reaches -3.5 dB on a
                    # partly-periodic source, because periodic components are
                    # cancellable despite the delay (E01/F3). Muting scores 0 dB
                    # and is strictly worse. This corrects the architecture's
                    # claim that an out-of-region controller always ADDS energy:
                    # it does so for unpredictable noise, not for periodic noise.
                    r = ear
            else:
                r = ear                                           # ipsilateral, always
            engage = True
            true_r, true_m = best_ref_for_ear(ear, th[i], TAU_S)
            if true_r is not None:
                ref_ok[ear] += 1
                if r == true_r:
                    ref_hit[ear] += 1
            # ---- filter choice ----
            w = pick_filter(bank, ear, r or ear, est) if filt_sel else bank["omni"]
            key = (r, None if not filt_sel else round(est / 30.0))
            if key != cur[ear]:
                cur[ear] = key
                anc[ear].set_filter(w, 256)
            anc[ear].engaged = engage
            engaged_frac[ear] += int(true_r is not None)
            ri = REFS.index(r) if r else 0
            blk = slice(i, i + FRAME)
            out = anc[ear].process_block_multiref(xs[:, blk],
                                                  (dL if ear == "L" else dR)[blk], ri)
            (eL if ear == "L" else eR)[blk] = out[0]

    m = slice(0, (n // FRAME) * FRAME)
    att = lambda e, d: 10 * np.log10((np.sum(e[m] ** 2) + 1e-30) /
                                     (np.sum(d[m] ** 2) + 1e-30))
    return dict(
        attL=att(eL, dL), attR=att(eR, dR),
        attJ=10 * np.log10((np.sum(eL[m] ** 2) + np.sum(eR[m] ** 2) + 1e-30) /
                           (np.sum(dL[m] ** 2) + np.sum(dR[m] ** 2) + 1e-30)),
        eL=eL, eR=eR, dL=dL, dR=dR, th=th, est=est_track[:nfr],
        rmse=float(np.sqrt(np.mean(np.square(est_src.err)))) if est_src.err else np.nan,
        conf=float(np.mean(est_src.conf)) if est_src.conf else np.nan,
        fallback=100.0 * est_src.tr.fallbacks / max(est_src.tr.n, 1),
        ref_acc={e: 100.0 * ref_hit[e] / max(ref_ok[e], 1) for e in "LR"},
        usable={e: 100.0 * ref_ok[e] / max(nfr, 1) for e in "LR"},
        engaged={e: 100.0 * engaged_frac[e] / max(nfr, 1) for e in "LR"},
        est_cost_ms=1e3 * est_src.cost_s / max(est_src.calls, 1),
        nfr=nfr)


# ---------------------------------------------------------------- metrics
def tracking_latency_ms(est, th, rate_dps):
    """Lag that best aligns the estimated bearing with the true one."""
    if len(est) < 8:
        return np.nan
    t_fr = np.arange(len(est)) * FRAME
    truth = th[np.clip(t_fr, 0, len(th) - 1)]
    a = est - est.mean(); b = truth - truth.mean()
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    n = len(a)
    lags = np.arange(-8, 9)
    best, bl = -2, 0
    for lg in lags:
        if lg >= 0:
            c = np.corrcoef(a[lg:], b[:n - lg])[0, 1] if n - lg > 3 else -2
        else:
            c = np.corrcoef(a[:n + lg], b[-lg:])[0, 1] if n + lg > 3 else -2
        if np.isfinite(c) and c > best:
            best, bl = c, lg
    return bl * FRAME / FS * 1e3


def att_after_direction_change(r, rate_dps, win_s=0.25):
    """Attenuation in the window following each sweep reversal."""
    th = r["th"]
    d_th = np.diff(np.sign(np.diff(th)))
    turns = np.where(np.abs(d_th) > 1)[0]
    if not len(turns):
        return np.nan
    num = den = 0.0
    k = int(win_s * FS)
    for t0 in turns:
        sl = slice(t0, min(t0 + k, len(th)))
        num += np.sum(r["eL"][sl] ** 2) + np.sum(r["eR"][sl] ** 2)
        den += np.sum(r["dL"][sl] ** 2) + np.sum(r["dR"][sl] ** 2)
    return 10 * np.log10((num + 1e-30) / (den + 1e-30))


def gcc_macs():
    nfft = 1 << int(np.ceil(np.log2(2 * FRAME)))
    gcc = 3 * 5 * nfft * np.log2(nfft)
    nseg = 6
    seg = int(2 * FRAME / (nseg + 1))
    nf2 = 1 << int(np.ceil(np.log2(seg)))
    coh = 6 * 2 * 5 * nf2 * np.log2(nf2)
    return gcc, coh


if __name__ == "__main__":
    t_start = time.time()
    rng = np.random.default_rng(9)
    s_plant = paths.add_electrical_delay(paths.secondary_path(FS, ntaps=1024),
                                         ELEC_DELAY, FS)
    cup = signal.butter(2, 1200.0 / (FS / 2), btype="low")

    print("E05 - direction and head rotation, two-cup headset\n")
    g = np.arange(-180, 180, 1.0)
    for ear in "LR":
        ipsi = np.mean([float(margin(ear, ear, t, TAU_S)) > 0 for t in g])
        print(f"  ear {ear}: usable reference over {ipsi*100:.0f}% of azimuth with the "
              f"ipsilateral mic only, {coverage(ear, TAU_S, g)*100:.0f}% using both")
    print()
    bank = build_bank(s_plant, cup, rng)

    RATES = [("slow", 30.0), ("medium", 100.0), ("fast", 250.0)]
    SNR = 20.0

    # ---------------- requirement 8: the 2x2 factorial, oracle bearing --------
    print("=" * 78)
    print("HYPOTHESIS TEST (oracle bearing, so the mechanism is isolated from")
    print("the quality of the bearing estimate)")
    print("=" * 78)
    print(f"{'config':<34}{'rate':>8}{'att L':>9}{'att R':>9}{'joint':>9}"
          f"{'after turn':>12}")
    print("-" * 78)
    fact = {}
    CFG = [("A  fixed ref + omni filter", (False, False, "oracle")),
           ("B  REF SELECTION + omni", (True, False, "oracle")),
           ("C  fixed ref + SECTOR FILTER", (False, True, "oracle")),
           ("D  both", (True, True, "oracle"))]
    for name, cfg in CFG:
        for rn, rd in RATES:
            r = run(cfg, rd, SNR, bank, s_plant, cup, rng)
            fact[(name[0], rn)] = r
            print(f"{name:<34}{rn:>8}{r['attL']:7.2f}dB{r['attR']:7.2f}dB"
                  f"{r['attJ']:7.2f}dB{att_after_direction_change(r, rd):10.2f}dB")
        print()

    mean = lambda ltr: float(np.mean([fact[(ltr, rn)]["attJ"] for rn, _ in RATES]))
    A, B, C, D = (mean(x) for x in "ABCD")
    print("-" * 78)
    print(f"mean joint attenuation:  A {A:6.2f}   B {B:6.2f}   C {C:6.2f}   D {D:6.2f}  dB")
    print(f"  reference selection alone (B-A) : {B-A:+6.2f} dB")
    print(f"  filter shape alone      (C-A)   : {C-A:+6.2f} dB")
    print(f"  both                    (D-A)   : {D-A:+6.2f} dB")
    hyp = (B - A) < -1.0 and abs(D - B) < abs(B - A) * 0.5
    print(f"\n  HYPOTHESIS "
          f"'{'the value of direction is causal reference selection, not filter shape'}'"
          f"\n  -> {'SUPPORTED' if hyp else 'NOT SUPPORTED'}")

    # ---------------- requirements 5-7: bearing-source ablation ---------------
    best_cfg = (True, (D - A) < (B - A) - 0.5, None)
    print("\n" + "=" * 78)
    print(f"BEARING-SOURCE ABLATION  (ref_select={best_cfg[0]}, "
          f"filter_select={best_cfg[1]})")
    print("=" * 78)
    print(f"{'bearing':<10}{'rate':>7}{'DoA RMSE':>10}{'lag ms':>8}{'ref acc':>9}"
          f"{'usable':>8}{'fallback':>10}{'joint att':>11}{'after turn':>12}")
    print("-" * 78)
    abl = {}
    for bmode in ("fixed", "audio", "imu", "fusion", "oracle"):
        for rn, rd in RATES:
            # The baseline arm must be config A -- NO direction at all. Running
            # the direction machinery with a constant bearing of zero is a
            # different (and much stronger) baseline, and using it hid a 2.4 dB
            # effect behind a 0.9 dB one on the first run.
            cfg = ((False, False, "fixed") if bmode == "fixed"
                   else (best_cfg[0], best_cfg[1], bmode))
            r = run(cfg, rd, SNR, bank, s_plant, cup, rng)
            abl[(bmode, rn)] = r
            acc = 0.5 * (r["ref_acc"]["L"] + r["ref_acc"]["R"])
            usable = 0.5 * (r["usable"]["L"] + r["usable"]["R"])
            lag = tracking_latency_ms(r["est"], r["th"], rd)
            print(f"{bmode:<10}{rn:>7}{r['rmse']:9.1f}d{lag:8.1f}{acc:8.0f}%"
                  f"{usable:7.0f}%{r['fallback']:9.0f}%{r['attJ']:9.2f}dB"
                  f"{att_after_direction_change(r, rd):10.2f}dB")
        print()

    # ---- gyro drift: the reason fusion exists, which a 3 s run cannot show ----
    print("-" * 78)
    print("GYRO DRIFT (bearing RMSE vs run length; gyro bias 0.5 deg/s)")
    print(f"  {'duration':>10}{'imu':>10}{'audio':>10}{'fusion':>10}")
    # NOTE: at module level a plain assignment IS the global. An earlier version
    # did `import rhear.experiments.e05_direction as _self; _self.DUR = d`, which
    # under `python -m`/script execution mutates a SECOND copy of the module --
    # so run() kept reading 3.0 and the drift table showed a gyro that never
    # drifted. It does: 8.7 deg RMSE at 30 s, reaching +15.2 deg.
    dur0 = DUR
    drift = {}
    for d in (3.0, 12.0, 30.0):
        DUR = d
        row = {}
        for bmode in ("imu", "audio", "fusion"):
            r = run((True, True, bmode), 30.0, SNR, bank, s_plant, cup, rng)
            row[bmode] = r["rmse"]
        drift[d] = row
        print(f"  {d:9.0f}s{row['imu']:9.1f}d{row['audio']:9.1f}d{row['fusion']:9.1f}d")
    DUR = dur0
    print()

    gcc, coh = gcc_macs()
    rate_hz = FS / FRAME
    print("-" * 78)
    print("computational cost of the bearing path, at 62.5 Hz:")
    print(f"  GCC-PHAT                {gcc*rate_hz/1e6:7.1f} MMAC/s")
    print(f"  coherence confidence    {coh*rate_hz/1e6:7.1f} MMAC/s")
    print(f"  gyro integration        {3*rate_hz/1e6:7.4f} MMAC/s")
    print(f"  audio total             {(gcc+coh)*rate_hz/1e6:7.1f} MMAC/s"
          f"   (L2 budget in the architecture doc is 15 MMAC/s)")
    for bmode in ("audio", "fusion"):
        c = np.mean([abl[(bmode, rn)]["est_cost_ms"] for rn, _ in RATES])
        print(f"  measured {bmode:<8}       {c:7.3f} ms/frame  "
              f"({100*c/(FRAME/FS*1e3):.1f}% of the 16 ms frame)")

    # ---------------- verdict -------------------------------------------------
    base = float(np.mean([abl[("fixed", rn)]["attJ"] for rn, _ in RATES]))
    assert abs(base - A) < 1.0, f"ablation baseline {base:.2f} should match config A {A:.2f}"
    res = {m: float(np.mean([abl[(m, rn)]["attJ"] for rn, _ in RATES]))
           for m in ("audio", "imu", "fusion", "oracle")}
    print("\n" + "=" * 78)
    print("G5 DECISION  (criterion: ANC improvement over the fixed baseline,")
    print("              NOT bearing accuracy)")
    print("=" * 78)
    print(f"  fixed baseline (no direction) : {base:6.2f} dB")
    for m in ("audio", "imu", "fusion", "oracle"):
        print(f"  {m:<30}: {res[m]:6.2f} dB   ({res[m]-base:+.2f} dB vs baseline)")
    realisable = max(res["audio"], res["imu"], res["fusion"])
    gain = base - realisable
    if not hyp and (base - res["oracle"]) < 1.0:
        verdict = "FAIL"
        why = ("neither mechanism produces an ANC gain even with a perfect "
               "bearing -- the directional mechanism should be removed")
    elif gain >= 1.0:
        verdict = "PASS"
        why = (f"a realisable bearing source improves ANC by {gain:.2f} dB over "
               f"the fixed baseline")
    elif (base - res["oracle"]) >= 1.0:
        verdict = "BLOCKED"
        why = (f"the mechanism is real (oracle {base-res['oracle']:+.2f} dB) but no "
               f"realisable estimator captures it (best {gain:+.2f} dB)")
    else:
        verdict = "FAIL"
        why = "no configuration improves ANC over the fixed baseline"
    print(f"\n  G5: {verdict}\n  {why}")
    print(f"\n({time.time()-t_start:.0f} s)")
    np.savez("results/e05_direction.npz",
             factorial={k: v["attJ"] for k, v in fact.items()},
             ablation={f"{m}_{r}": abl[(m, r)]["attJ"] for m, r in abl},
             verdict=verdict, hypothesis_supported=hyp,
             A=A, B=B, C=C, D=D, baseline=base, **{f"res_{k}": v for k, v in res.items()})
