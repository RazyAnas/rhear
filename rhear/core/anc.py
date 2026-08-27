"""L0 control: FxNLMS family with robust score functions and impulse gating."""
import numpy as np
from scipy import signal


def score(e, mode, eps=1e-6, p=1.2):
    """psi(e) in  w <- w + mu * xhat * psi(e) / (delta + ||xhat||^2)."""
    if mode == "nlms":
        return e
    if mode == "log":                       # FxlogLMS: stable without knowing alpha
        return e / (eps + abs(e))
    if mode == "p_power":                   # FxLMP, needs p < alpha
        return np.sign(e) * abs(e) ** (p - 1.0)
    raise ValueError(mode)


class ImpulseDetector:
    """Sample-domain transient detector: onset slope + crest factor, hysteretic."""

    def __init__(self, fs, win=64, r_th_db=15.0, c_th=6.0, hold_s=0.20):
        self.win, self.r_th, self.c_th = win, r_th_db, c_th
        self.hold = int(hold_s * fs)
        self.buf = np.zeros(2 * win)
        self.i = 0
        self.countdown = 0

    def __call__(self, x):
        self.buf[self.i % (2 * self.win)] = x
        self.i += 1
        if self.countdown > 0:
            self.countdown -= 1
            return True
        if self.i < 2 * self.win:
            return False
        idx = np.arange(self.i - 2 * self.win, self.i) % (2 * self.win)
        recent, older = self.buf[idx[self.win:]], self.buf[idx[:self.win]]
        e_now, e_prev = np.sum(recent ** 2) + 1e-20, np.sum(older ** 2) + 1e-20
        r_db = 10 * np.log10(e_now / e_prev)
        crest = np.max(np.abs(recent)) / (np.sqrt(e_now / self.win) + 1e-20)
        if r_db > self.r_th and crest > self.c_th:
            self.countdown = self.hold
            return True
        return False


def fxnlms(x, d, s_true, s_hat, L=256, mu=0.05, delta=1e-4, leak=0.0,
           mode="nlms", freeze_on_impulse=False, fs=48_000, track_every=1,
           normalize=True):
    """Run a feedforward ANC loop.

    x       reference signal (what the reference mic hears)
    d       disturbance at the error mic (primary path output)
    s_true  true secondary path, INCLUDING all electrical delay
    s_hat   the controller's model of it
    Returns dict with e (residual), w_norm trace, and frozen flag trace.
    """
    n = len(x)
    xhat = signal.lfilter(s_hat, [1.0], x)          # filtered reference
    w = np.zeros(L)
    y = np.zeros(n)
    e = np.zeros(n)
    w_norm = np.zeros(n // track_every + 1)
    frozen = np.zeros(n, dtype=bool)
    det = ImpulseDetector(fs) if freeze_on_impulse else None

    ls = len(s_true)
    xbuf = np.zeros(L)
    xhbuf = np.zeros(L)
    ybuf = np.zeros(ls)

    d_rms = np.sqrt(np.mean(d ** 2)) + 1e-20
    diverged = False
    for i in range(n):
        xbuf[1:] = xbuf[:-1]; xbuf[0] = x[i]
        xhbuf[1:] = xhbuf[:-1]; xhbuf[0] = xhat[i]

        yi = float(w @ xbuf)
        ybuf[1:] = ybuf[:-1]; ybuf[0] = yi
        y[i] = yi
        # ybuf[0] is the newest sample, so conv(s,y)[i] == s_true . ybuf
        e[i] = d[i] - float(s_true @ ybuf)

        hold = det(x[i]) if det is not None else False
        frozen[i] = hold
        if not hold:
            # Normalisation is itself an impulse defence: dividing by ||xhat||^2
            # divides out a loud reference. Unnormalised FxLMS has no such
            # protection, which is the configuration the divergence result is about.
            nrm = (delta + float(xhbuf @ xhbuf)) if normalize else 1.0
            w *= (1.0 - leak)
            w += (mu / nrm) * xhbuf * score(e[i], mode)
        if i % track_every == 0:
            w_norm[i // track_every] = np.linalg.norm(w)
            if abs(e[i]) > 1e4 * d_rms:          # unmistakably blown up
                diverged = True
                e[i:] = np.nan
                break
    return dict(e=e, y=y, w=w, w_norm=w_norm, frozen=frozen, diverged=diverged)


def nmse_db(e, d, skip=0.5):
    """Residual power relative to disturbance power, over the last (1-skip) fraction.

    Returns +inf for a diverged run rather than a meaningless large number.
    """
    k = int(len(e) * skip)
    seg = e[k:]
    if not np.all(np.isfinite(seg)):
        return np.inf
    num = float(np.sum(seg.astype(np.float64) ** 2))
    den = float(np.sum(d[k:].astype(np.float64) ** 2)) + 1e-30
    if not np.isfinite(num) or num / den > 1e6:
        return np.inf
    return 10 * np.log10(num / den + 1e-30)
