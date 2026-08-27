"""Head geometry with a moving source / rotating listener.

Plane wave from azimuth theta (0 = straight ahead, +ve to the right). Arrival
time at a point p on the head, relative to the head centre, is -(p.u)/c.

The important consequence, and the reason direction is a control variable rather
than a perception feature: the causality margin for a cup is

    margin(theta) = [ d_fwd*cos(theta) + d_out*sin(theta) ] / c  -  tau_secondary

so a source ahead gives the largest margin and a source behind gives a NEGATIVE
one -- the wavefront reaches the ear before the reference mic has seen it, and
feedforward control is impossible in that band no matter how fast the electronics.
"""
import numpy as np

C_AIR = 343.0
HEAD_R = 0.0875          # sphere radius, m
D_FWD = 0.050            # reference mic forward of the ear, m
D_OUT = 0.015            # reference mic outboard of the ear, m


def _tau(p, theta):
    """Arrival delay at point p = (x_fwd, y_left) for azimuth theta (radians)."""
    u = np.array([np.cos(theta), -np.sin(theta)])      # +theta is to the right
    return -(p[0] * u[0] + p[1] * u[1]) / C_AIR


def delays(theta_rad):
    """(tau_refL, tau_refR, tau_earL) for one azimuth, in seconds (relative)."""
    earL = np.array([0.0, +HEAD_R])
    refL = np.array([D_FWD, +HEAD_R + D_OUT])
    refR = np.array([D_FWD, -HEAD_R - D_OUT])
    return _tau(refL, theta_rad), _tau(refR, theta_rad), _tau(earL, theta_rad)


# --- the full reference -> ear matrix -------------------------------------
# A headset has two ears and two reference microphones, so there are FOUR
# feedforward pairings, not two. Ignoring the contralateral pairings throws away
# most of the angular coverage: for a source at +90 deg the left cup's own
# reference is 102 us non-causal, while the RIGHT reference leads the left ear by
# roughly half a head width and is comfortably causal.
POINTS = {
    "earL": (0.0, +HEAD_R),
    "earR": (0.0, -HEAD_R),
    "refL": (D_FWD, +(HEAD_R + D_OUT)),
    "refR": (D_FWD, -(HEAD_R + D_OUT)),
}


def tau_point(name, theta_deg):
    """Plane-wave arrival delay at a named point, seconds (relative to centre)."""
    return _tau(POINTS[name], np.radians(np.asarray(theta_deg, dtype=float)))


def margin(ref, ear, theta_deg, tau_secondary):
    """Causality margin for driving `ear` from `ref`, seconds.

        margin = tau_ear(theta) - tau_ref(theta) - tau_secondary

    Positive means a causal feedforward controller is possible at that azimuth.
    """
    return (tau_point("ear" + ear, theta_deg) - tau_point("ref" + ref, theta_deg)
            - tau_secondary)


def best_ref_for_ear(ear, theta_deg, tau_secondary):
    """Which reference gives this ear the largest causality margin?
    Returns (ref, margin). ref is None when neither is causal."""
    mL = float(margin("L", ear, theta_deg, tau_secondary))
    mR = float(margin("R", ear, theta_deg, tau_secondary))
    if max(mL, mR) <= 0:
        return None, max(mL, mR)
    return ("L", mL) if mL >= mR else ("R", mR)


def coverage(ear, tau_secondary, grid=None):
    """Fraction of azimuth for which this ear has ANY usable reference."""
    g = grid if grid is not None else np.arange(-180, 180, 1.0)
    ok = [best_ref_for_ear(ear, t, tau_secondary)[0] is not None for t in g]
    return float(np.mean(ok))


def causality_margin(theta_deg, tau_secondary, cup="L"):
    """Electrical-delay budget available at this azimuth, for one cup, seconds.

    Derived from tau_ear - tau_ref for that cup, so it agrees with delays():

        left  cup:  [ D_FWD*cos(th) - D_OUT*sin(th) ] / c  -  tau_secondary
        right cup:  [ D_FWD*cos(th) + D_OUT*sin(th) ] / c  -  tau_secondary

    The two cups therefore have DIFFERENT causal cones, and each covers azimuths
    the other cannot. A first version of this function used the right-cup sign for
    both, which overstated the left cup's coverage and quietly made part of an
    experiment's sweep non-causal.
    """
    th = np.radians(np.asarray(theta_deg, dtype=float))
    sgn = -1.0 if cup.upper() == "L" else +1.0
    return (D_FWD * np.cos(th) + sgn * D_OUT * np.sin(th)) / C_AIR - tau_secondary


def causal_cone(tau_secondary, cup="L", grid=None):
    """Azimuth range over which this cup's reference mic can be used at all."""
    g = grid if grid is not None else np.arange(-180, 181, 1.0)
    m = causality_margin(g, tau_secondary, cup)
    ok = g[m > 0]
    return (ok.min(), ok.max()) if len(ok) else (np.nan, np.nan)


def mic_spacing():
    return 2 * (HEAD_R + D_OUT)


def theta_from_itd(itd_s):
    """Far-field azimuth from inter-reference delay. Returns the FRONT solution;
    the back solution is 180 - theta (the cone of confusion), which two mics on a
    head cannot separate without motion or a third non-collinear sensor."""
    s = np.clip(itd_s * C_AIR / mic_spacing(), -1.0, 1.0)
    return np.degrees(np.arcsin(s))


def render(src, theta_traj, fs):
    """Render a source through a time-varying geometry.

    theta_traj : per-sample azimuth in degrees (head rotation or source motion).
    Returns (x_refL, x_refR, d_ear) with fractional, time-varying delays applied
    by linear interpolation into the source buffer.
    """
    n = len(src)
    th = np.radians(theta_traj)
    tL, tR, tE = delays(th)
    base = max(np.max(tL), np.max(tR), np.max(tE)) + 1e-3      # keep reads causal
    out = []
    idx = np.arange(n)
    for tau in (tL, tR, tE):
        pos = idx - (tau + base) * fs
        i0 = np.floor(pos).astype(int)
        frac = pos - i0
        i0c = np.clip(i0, 0, n - 2)
        y = (1 - frac) * src[i0c] + frac * src[i0c + 1]
        y[(i0 < 0) | (i0 >= n - 1)] = 0.0
        out.append(y)
    return out
