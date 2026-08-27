"""Unit tests for the head geometry and per-cup causality (G5 requirement 2).

Each margin is checked against an INDEPENDENT closed-form expression derived by
hand from the plane-wave arrival times, not against the implementation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rhear.sim.geometry import (margin, tau_point, best_ref_for_ear, coverage,
                                C_AIR, HEAD_R, D_FWD, D_OUT)

TAU_S = 0.02 / C_AIR
A, F, O = HEAD_R, D_FWD, D_OUT


def analytic(ref, ear, th_deg):
    """Closed form, derived by hand:  tau(p) = (-p_x cos th + p_y sin th)/c
    with +theta to the right, so p_y is +A for the left ear."""
    t = np.radians(th_deg)
    y_ear = +A if ear == "L" else -A
    y_ref = +(A + O) if ref == "L" else -(A + O)
    tau_ear = (0.0 * -np.cos(t) + y_ear * np.sin(t)) / C_AIR
    tau_ref = (-F * np.cos(t) + y_ref * np.sin(t)) / C_AIR
    return tau_ear - tau_ref - TAU_S


def test_margins_match_closed_form():
    bad = []
    for ref in "LR":
        for ear in "LR":
            for th in range(-180, 181, 5):
                a = analytic(ref, ear, th)
                b = float(margin(ref, ear, th, TAU_S))
                if abs(a - b) > 1e-12:
                    bad.append((ref, ear, th, a, b))
    assert not bad, f"{len(bad)} mismatches, first {bad[:3]}"
    return len(range(-180, 181, 5)) * 4


def test_symmetry():
    """Mirroring the source must mirror left and right."""
    for th in range(0, 181, 5):
        assert abs(float(margin("L", "L", -th, TAU_S)) -
                   float(margin("R", "R", +th, TAU_S))) < 1e-12
        assert abs(float(margin("R", "L", -th, TAU_S)) -
                   float(margin("L", "R", +th, TAU_S))) < 1e-12


def test_front_is_best_for_ipsilateral():
    """Straight ahead, both ipsilateral margins equal F/c - tau_s."""
    for ear in "LR":
        m = float(margin(ear, ear, 0.0, TAU_S))
        assert abs(m - (F / C_AIR - TAU_S)) < 1e-12


def test_behind_is_never_causal():
    """From directly behind the wavefront reaches the ear first, whatever the
    reference: the mic is forward of the ear and the geometry is symmetric."""
    for ref in "LR":
        for ear in "LR":
            assert float(margin(ref, ear, 180.0, TAU_S)) < 0


def test_contralateral_extends_coverage():
    """The whole point: using both references covers far more azimuth than one."""
    g = np.arange(-180, 180, 1.0)
    ipsi_only = np.mean([float(margin("L", "L", t, TAU_S)) > 0 for t in g])
    both = coverage("L", TAU_S, g)
    assert both > ipsi_only + 0.15, f"ipsi {ipsi_only:.3f} both {both:.3f}"
    return ipsi_only, both


def test_best_ref_picks_the_larger_margin():
    for th in range(-180, 181, 3):
        ref, m = best_ref_for_ear("L", th, TAU_S)
        mL = float(margin("L", "L", th, TAU_S))
        mR = float(margin("R", "L", th, TAU_S))
        if max(mL, mR) <= 0:
            assert ref is None
        else:
            assert ref == ("L" if mL >= mR else "R")
            assert abs(m - max(mL, mR)) < 1e-12


if __name__ == "__main__":
    n = test_margins_match_closed_form()
    print(f"  margins vs closed form ........ PASS ({n} cases, |err| < 1e-12)")
    test_symmetry();            print("  left/right mirror symmetry .... PASS")
    test_front_is_best_for_ipsilateral(); print("  straight ahead = F/c - tau_s .. PASS")
    test_behind_is_never_causal();        print("  behind never causal ........... PASS")
    i, b = test_contralateral_extends_coverage()
    print(f"  contralateral extends cover ... PASS ({i*100:.0f}% -> {b*100:.0f}% of azimuth)")
    test_best_ref_picks_larger = test_best_ref_picks_the_larger_margin()
    print("  best_ref picks larger margin .. PASS")
    print("\n  reference -> ear margins (us):")
    print(f"  {'azimuth':>8}{'L->L':>9}{'R->L':>9}{'L->R':>9}{'R->R':>9}   best for L ear")
    for th in (-150,-120,-90,-60,-30,0,30,60,90,120,150,180):
        r,_ = best_ref_for_ear("L", th, TAU_S)
        vals = [float(margin(a,b,th,TAU_S))*1e6 for a,b in (("L","L"),("R","L"),("L","R"),("R","R"))]
        print(f"  {th:7.0f}"+"".join(f"{v:9.1f}" for v in vals)+
              f"   {r or 'NONE - disengage'}")
