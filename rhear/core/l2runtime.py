"""L0 runtime with a frame-rate coefficient interface.

This is the mechanism the whole architecture rests on: a slow selector writes
control-filter coefficients at frame rate (62.5 Hz), while the fast loop runs
FxNLMS per sample. The selector never touches audio.
"""
import numpy as np
from scipy import signal
from .anc import score


def run_controller(x, d, s_true, s_hat, L=256, mu=0.01, delta=1e-4, leak=0.0,
                   mode="nlms", w_init=None, selector=None, frame=256,
                   refine=True, xfade=0.3, fs=48_000):
    """Run the fast loop.

    selector : callable(frame_index, x_frame_history) -> w0 (length L) or None
               Called at each frame boundary. Returning None leaves w alone.
    refine   : if False, coefficients are frozen between selector calls
               (pure fixed-filter operation, no adaptation at all).
    xfade    : fraction of a frame over which to blend to a newly selected w0,
               so a switch does not click.
    """
    n = len(x)
    xhat = signal.lfilter(s_hat, [1.0], x)
    w = np.zeros(L) if w_init is None else w_init.astype(float).copy()
    e = np.zeros(n)
    ls = len(s_true)
    xbuf, xhbuf, ybuf = np.zeros(L), np.zeros(L), np.zeros(ls)
    w_target, blend_left, blend_n = None, 0, max(1, int(xfade * frame))
    d_rms = np.sqrt(np.mean(d ** 2)) + 1e-20

    for i in range(n):
        if selector is not None and i % frame == 0:
            w0 = selector(i // frame, x[:i])
            if w0 is not None:
                w_target, blend_left = w0.astype(float), blend_n
        if blend_left > 0:                       # linear cross-fade to the new filter
            a = 1.0 / blend_left
            w = (1 - a) * w + a * w_target
            blend_left -= 1

        xbuf[1:] = xbuf[:-1]; xbuf[0] = x[i]
        xhbuf[1:] = xhbuf[:-1]; xhbuf[0] = xhat[i]
        yi = float(w @ xbuf)
        ybuf[1:] = ybuf[:-1]; ybuf[0] = yi
        e[i] = d[i] - float(s_true @ ybuf)

        if refine and blend_left == 0:
            nrm = delta + float(xhbuf @ xhbuf)
            w *= (1.0 - leak)
            w += (mu / nrm) * xhbuf * score(e[i], mode)
        if i % 512 == 0 and abs(e[i]) > 1e4 * d_rms:
            e[i:] = np.nan
            break
    return dict(e=e, w=w)


def train_filter(x, d, s, L=256, mu=0.01, passes=3, mode="nlms"):
    """Converge a control filter on one noise type. This is how the pre-trained
    fixed-filter bank is built -- exactly the SFANC/GFANC premise."""
    w = np.zeros(L)
    for _ in range(passes):
        r = run_controller(x, d, s, s, L=L, mu=mu, mode=mode, w_init=w,
                           selector=None, refine=True)
        if not np.all(np.isfinite(r["e"])):
            mu *= 0.3
            w = np.zeros(L)
            continue
        w = r["w"]
    return w


def sliding_nmse_db(e, d, fs, win_s=0.05):
    """Attenuation as a function of time."""
    w = int(win_s * fs)
    k = len(e) // w
    out = np.full(k, np.nan)
    for i in range(k):
        a, b = i * w, (i + 1) * w
        seg = e[a:b]
        if np.all(np.isfinite(seg)):
            out[i] = 10 * np.log10((np.sum(seg ** 2) + 1e-30) /
                                   (np.sum(d[a:b] ** 2) + 1e-30))
    return out
