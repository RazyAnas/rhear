"""E01 - The codec decision, proved in simulation.

Sweeps total electrical delay through the ANC loop and measures attenuation for
three noise characters spanning the predictability axis. Tests four claims:

  C1  A causality cliff exists and sits where theory says: tau_p - tau_s.
  C2  The cliff position depends on PREDICTABILITY, not on loudness or band --
      which is why head A regresses H (architecture 5.2) rather than a class label.
  C3  Achieved attenuation tracks the D-step prediction floor, so the simulator
      explains the cliff rather than merely displaying it.
  C4  Part selection: 5 us ANC codec is inside the budget, a 619 us generic codec
      is not, and a laptop over USB (~6 ms) is not remotely close.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy import signal
from rhear.sim import paths
from rhear.sim.electronics import decimation_chain_group_delay
from rhear.core import signals as sg
from rhear.core.anc import fxnlms, nmse_db
from rhear.core.predict import prediction_floor_db

FS = 192_000
DUR = 0.35
MIC_DIST, DRV_DIST = 0.07, 0.02
MU_LADDER = (0.02, 0.01, 0.003, 0.001, 3e-4)
DELAYS_US = [0, 5, 25, 60, 100, 146, 200, 300, 450, 619, 1000]


def run_case(x, L, label, leak=0.0):
    """For each delay take the BEST result over the step-size ladder: a real
    system tunes mu per condition (architecture head C), so comparing at a single
    fixed mu would confuse a step-size problem with a causality problem."""
    p = paths.primary_path(FS, distance_m=MIC_DIST)
    s_ac = paths.secondary_path(FS, distance_m=DRV_DIST)
    d = signal.lfilter(p, [1.0], x)
    nmse, best_mus, floors = [], [], []
    for du in DELAYS_US:
        s_tot = paths.add_electrical_delay(s_ac, du * 1e-6, FS)
        best, best_mu = np.inf, np.nan
        for mu in MU_LADDER:
            r = fxnlms(x, d, s_tot, s_tot, L=L, mu=mu, leak=leak,
                       mode="nlms", fs=FS, track_every=64)
            v = nmse_db(r["e"], d, skip=0.6)
            if np.isfinite(v) and v < best:
                best, best_mu = v, mu
        # theoretical floor: total loop delay in samples, seen by the controller
        D = (du * 1e-6 + DRV_DIST / paths.C_AIR - MIC_DIST / paths.C_AIR) * FS
        floor = prediction_floor_db(x, max(D, 0), order=256) if D > 0 else -np.inf
        nmse.append(best); best_mus.append(best_mu); floors.append(floor)
        fl = f"{floor:7.1f}" if np.isfinite(floor) else "   caus"
        print(f"    {label:12s} tau_e={du:6.1f} us   NMSE={best:7.2f} dB   "
              f"floor={fl} dB   (mu={best_mu:g})", flush=True)
    return np.array(nmse), np.array(best_mus), np.array(floors)


if __name__ == "__main__":
    tau_p, tau_s = MIC_DIST / paths.C_AIR, DRV_DIST / paths.C_AIR
    budget = tau_p - tau_s
    gen = decimation_chain_group_delay()
    n = int(FS * DUR)

    print(f"Geometry: reference mic {MIC_DIST*100:.0f} cm, driver {DRV_DIST*100:.0f} cm from ear")
    print(f"  tau_primary = {tau_p*1e6:.1f} us   tau_secondary = {tau_s*1e6:.1f} us")
    print(f"  CAUSALITY BUDGET for the electrical chain = {budget*1e6:.1f} us\n")

    cases = [
        ("harmonic",  sg.harmonic(n, FS, f0=200.0, n_harm=8),          1600, 1e-6),
        ("bandlimited", sg.broadband(n, FS, 80, 2000),                  512, 0.0),
        ("wideband",  sg.broadband(n, FS, 200, 8000),                   512, 0.0),
    ]
    print("  predictability H at the 146 us budget:")
    for name, x, _, _ in cases:
        from rhear.core.predict import predictability
        print(f"    {name:12s} H = {predictability(x, budget*FS):.4f}")
    print()

    t0, res = time.time(), {}
    for name, x, L, leak in cases:
        res[name] = run_case(x, L, name, leak=leak)
        print()
    print(f"  ({time.time()-t0:.0f} s)\n")

    print("=" * 86)
    print(f"{'part':<32}{'delay':>10}" + "".join(f"{k:>14}" for k, _, _, _ in cases))
    print("-" * 86)
    for pname, dus in [("analog ANC IC (AS3415)", 0.0),
                       ("ADAU1777/1787 ANC codec", 5.0),
                       ("generic 48 kHz codec", gen * 1e6),
                       ("laptop over USB audio", 6000.0)]:
        row = f"{pname:<32}{dus:8.0f} us"
        for k, _, _, _ in cases:
            v = np.interp(dus, DELAYS_US, res[k][0]) if dus <= DELAYS_US[-1] else res[k][0][-1]
            row += f"{v:11.1f} dB"
        if dus > DELAYS_US[-1]:
            row += "  (extrapolated)"
        print(row)
    print("=" * 86)
    np.savez("results/e01_causality.npz", delays=DELAYS_US, budget_us=budget * 1e6,
             generic_codec_us=gen * 1e6,
             **{f"{k}_{f}": res[k][i] for k in res for i, f in enumerate(("nmse", "mu", "floor"))})
    print("saved -> results/e01_causality.npz")
