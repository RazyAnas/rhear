"""Streaming L0 controller -- the same update rule as rhear.core.anc, but
stateful across blocks so it can run live.

The controller half of this class is what ports to the MCU: it consumes a
reference sample and a measured error sample and emits an anti-noise sample. The
plant half (applying the true secondary path to produce the error) exists only in
simulation; on hardware the error arrives from the error microphone instead.
"""
import numpy as np
from .anc import score


class StreamingANC:
    def __init__(self, L, mu, s_hat, s_true=None, fs=48_000, mode="nlms",
                 delta=1e-4, leak=0.0):
        self.L, self.mu, self.delta, self.leak, self.mode = L, mu, delta, leak, mode
        self.fs = fs
        self.s_hat = np.asarray(s_hat, float)
        self.s_true = None if s_true is None else np.asarray(s_true, float)
        self.w = np.zeros(L)
        self.xbuf = np.zeros(L)
        self.xhbuf = np.zeros(L)
        self.ybuf = np.zeros(len(self.s_true) if self.s_true is not None else 1)
        self.shbuf = np.zeros(len(self.s_hat))
        self.engaged = True
        self.adapting = True
        self._target = None
        self._blend = 0
        self._blend_n = 1
        self._mbuf = self._mhbuf = self._msh = None

    def set_filter(self, w0, xfade_samples=256):
        """The L2 coefficient path. Cross-faded so a switch does not click."""
        self._target = np.asarray(w0, float).copy()
        self._blend_n = max(1, int(xfade_samples))
        self._blend = self._blend_n

    def process_block(self, x, d):
        """x: reference samples. d: disturbance at the error mic (simulation).
        Returns (e, y)."""
        n = len(x)
        e = np.empty(n)
        y = np.empty(n)
        w, xbuf, xhbuf, ybuf, shbuf = (self.w, self.xbuf, self.xhbuf,
                                       self.ybuf, self.shbuf)
        s_hat, s_true = self.s_hat, self.s_true
        for i in range(n):
            if self._blend > 0:
                a = 1.0 / self._blend
                w *= (1 - a); w += a * self._target
                self._blend -= 1
            xbuf[1:] = xbuf[:-1]; xbuf[0] = x[i]
            shbuf[1:] = shbuf[:-1]; shbuf[0] = x[i]
            xhbuf[1:] = xhbuf[:-1]; xhbuf[0] = float(s_hat @ shbuf)
            yi = float(w @ xbuf) if self.engaged else 0.0
            ybuf[1:] = ybuf[:-1]; ybuf[0] = yi
            ei = d[i] - float(s_true @ ybuf)
            e[i] = ei; y[i] = yi
            if self.adapting and self.engaged and self._blend == 0:
                nrm = self.delta + float(xhbuf @ xhbuf)
                if self.leak:
                    w *= (1.0 - self.leak)
                w += (self.mu / nrm) * xhbuf * score(ei, self.mode)
        self.w = w
        return e, y

    def process_block_multiref(self, xs, d, ref):
        """Same loop, but with rolling buffers kept for EVERY reference mic so a
        reference switch does not inherit a stale delay line.

        A headset has two reference mics and two ears, giving four feedforward
        pairings; which one is causal depends on azimuth, so the controller must
        be able to change reference without a transient.
        """
        xs = np.atleast_2d(xs)
        nref, n = xs.shape
        if self._mbuf is None or self._mbuf.shape[0] != nref:
            self._mbuf = np.zeros((nref, self.L))
            self._mhbuf = np.zeros((nref, self.L))
            self._msh = np.zeros((nref, len(self.s_hat)))
        e = np.empty(n); y = np.empty(n)
        w = self.w
        xb, xhb, shb = self._mbuf, self._mhbuf, self._msh
        ybuf, s_hat, s_true = self.ybuf, self.s_hat, self.s_true
        for i in range(n):
            if self._blend > 0:
                a = 1.0 / self._blend
                w *= (1 - a); w += a * self._target
                self._blend -= 1
            xb[:, 1:] = xb[:, :-1]; xb[:, 0] = xs[:, i]
            shb[:, 1:] = shb[:, :-1]; shb[:, 0] = xs[:, i]
            xhb[:, 1:] = xhb[:, :-1]; xhb[:, 0] = shb @ s_hat
            yi = float(w @ xb[ref]) if self.engaged else 0.0
            ybuf[1:] = ybuf[:-1]; ybuf[0] = yi
            ei = d[i] - float(s_true @ ybuf)
            e[i] = ei; y[i] = yi
            if self.adapting and self.engaged and self._blend == 0:
                xh = xhb[ref]
                nrm = self.delta + float(xh @ xh)
                if self.leak:
                    w *= (1.0 - self.leak)
                w += (self.mu / nrm) * xh * score(ei, self.mode)
        self.w = w
        return e, y

    @property
    def w_norm(self):
        return float(np.linalg.norm(self.w))
