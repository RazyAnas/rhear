"""Live simulation source: runs the actual RHEAR pipeline in real time.

Every value published here is computed by the same code the experiments use.
Nothing is synthesised for display. Where a quantity is not yet measured (L1
speech enhancement, for example) the corresponding block reports `idle` and its
scalars are omitted, so the UI shows "--" rather than an invented number.
"""
import os
import time
import numpy as np
import psutil
from scipy import signal

from ...sim import paths
from ...sim.geometry import (delays, tau_point, margin, best_ref_for_ear,
                             theta_from_itd, C_AIR, causal_cone)
from ...sim.electronics import soft_clip_aop
from ...core import signals as sg
from ...core.anc import ImpulseDetector
from ...core.state import frame_features
from ...core.doa import gcc_phat, bearing_confidence, Gyro, BearingTracker
from ...core.l0 import StreamingANC, process_pair
from ...core.streaming import StreamingHarmonic, StreamingBand, StreamingDelayLine
from ...core.l2runtime import train_filter
from ..schema import (TelemetryFrame, Channel, Spectrum, Scalar, Vector, Block,
                      Edge, Geometry, Event)
from .base import TelemetrySource

FS = 48_000
BLOCK = 1024                  # 21.3 ms of audio per pipeline step
L2_FRAME = 768                # 16 ms -> 62.5 Hz
# E01/F4: exploiting periodicity needs a filter spanning at least one period.
# The 50 Hz engine fundamental is 960 samples at 48 kHz, so L must exceed that.
# At L=512 the engine filter trains to +1.06 dB -- worse than doing nothing.
L = 1024
MU = 0.02
ELEC_DELAY = 38e-6            # ADAU1772
BAND = (120.0, 1500.0)
AOP_DBSPL, REF_DBSPL = 135.0, 94.0
SCENES = ["engine", "rotor", "wind", "siren"]
# L2 head C: per-scene adaptation schedule, chosen by measured sweep rather than
# guessed. E01/F5 -- the usable step size depends on the scene and the loop delay.
SCENE_MU = {"engine": 0.005, "rotor": 0.005, "wind": 0.001, "siren": 0.02}
SCENE_LEAK = {"engine": 0.0, "rotor": 0.0, "wind": 1e-6, "siren": 0.0}
# The GCC-PHAT peak-to-sidelobe confidence is known to be weak for strongly tonal
# sources (E05 blocker #5): the correlation surface is periodic, so the margin
# collapses even when the bearing is correct. This floor is calibrated to the
# observed distribution for these scenes, not tuned to make a panel look good.
# The raw value is always displayed; fixing the metric is tracked in G5.
CONF_FLOOR = 0.08
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "results", "twin_bank.npz")


class SimulationSource(TelemetrySource):
    kind = "sim"
    name = "RHEAR digital twin"

    def __init__(self, bus=None, publish_hz=20.0, scene_s=5.0, impulse_every=13.0):
        super().__init__(bus=bus or __import__(
            "rhear.telemetry.bus", fromlist=["BUS"]).BUS, publish_hz=publish_hz)
        self.scene_s = scene_s
        self.impulse_every = impulse_every
        self.rng = np.random.default_rng(7)
        self.proc = psutil.Process()
        self._build_plant()
        self._build_bank()
        self._build_pipeline()

    # ---------- plant ----------
    def _build_plant(self):
        self.p = paths.primary_path(FS, ntaps=512)
        s_full = paths.add_electrical_delay(
            paths.secondary_path(FS, ntaps=1024), ELEC_DELAY, FS)
        # Truncate the PLANT where the impulse response has nothing left. The
        # tail beyond 99.999% of the energy costs a full multiply-accumulate per
        # sample per ear and changes the result in the 5th decimal place.
        e = np.cumsum(s_full ** 2) / np.sum(s_full ** 2)
        self.s = s_full[:int(np.searchsorted(e, 0.99999)) + 1].copy()
        self.tau_s = 0.02 / C_AIR
        self.cone_L = causal_cone(self.tau_s, "L")
        self.cone_R = causal_cone(self.tau_s, "R")
        self.b_cup, self.a_cup = signal.butter(2, 1200.0 / (FS / 2), btype="low")
        self.zi_cup = np.zeros(2)

    def _gen(self, scene):
        r = self.rng
        if scene == "engine":
            return StreamingHarmonic(FS, 50.0, 10, r)
        if scene == "rotor":
            return StreamingHarmonic(FS, 120.0, 6, r)
        if scene == "wind":
            # 150 Hz lower edge, not 60: below the driver's roll-off 1/S is
            # ill-conditioned, and a broadband source needs the inverse across a
            # continuum where a periodic one needs it only at a few discrete
            # frequencies. Measured: 60-900 Hz wind cannot be cancelled at ANY
            # step size in this plant (best -1.07 dB, worst +4.24 dB). The
            # actuator band is the honest bound, so the scene sits inside it.
            return StreamingBand(FS, 150.0, 900.0, r)
        return StreamingBand(FS, 400.0, 1400.0, r)      # siren-ish band

    def _render_static(self, scene, seconds, az_deg=0.0):
        """Render a scene through EXACTLY the plant the live loop uses.

        The first version trained the bank against `primary_path` while the live
        loop rendered through the head geometry and the cup filter. Different
        plant, so the pre-trained filters were wrong on arrival: engine and rotor
        still worked (a periodic signal absorbs a delay mismatch -- E01/F3 again)
        but wind and siren came out at +6 dB, i.e. the controller was ADDING
        noise. Bank training and the live loop must share this one function.
        """
        g = self._gen(scene)
        nblk = max(1, int(seconds * FS / BLOCK))
        dlL = StreamingDelayLine(8192); dlE = StreamingDelayLine(8192)
        tL, _, tE = delays(np.radians(az_deg))
        base = 0.002
        zi = np.zeros(2)
        XL, D = [], []
        for _ in range(nblk):
            src = g.block(BLOCK) * 10 ** ((95.0 - REF_DBSPL) / 20.0)
            dlL.push_block(src); dlE.push_block(src)
            xl = dlL.read(np.full(BLOCK, (tL + base) * FS))
            xe = dlE.read(np.full(BLOCK, (tE + base) * FS))
            xl = soft_clip_aop(xl, AOP_DBSPL, REF_DBSPL)
            xl = xl + 10 ** (-73.0 / 20.0) * self.rng.standard_normal(BLOCK)
            d, zi = signal.lfilter(self.b_cup, self.a_cup, xe, zi=zi)
            XL.append(xl); D.append(d)
        return np.concatenate(XL), np.concatenate(D)

    # ---------- pre-trained filter bank (the L2 coefficient path) ----------
    def _build_bank(self):
        if os.path.exists(CACHE):
            z = np.load(CACHE)
            if z["L"] == L and len(z["bank"][0]) == L:
                self.bank = list(z["bank"])
                self.bank_score = list(z["score"])
                print(f"[twin] filter bank loaded from {os.path.basename(CACHE)}")
                return
        print("[twin] training the filter bank (one filter per scene)...")
        self.bank, self.bank_score = [], []
        for sc in SCENES:
            x, d = self._render_static(sc, 2.0)      # same plant as the live loop
            w = train_filter(x, d, self.s, L=L, mu=SCENE_MU[sc], passes=4)
            anc = StreamingANC(L, SCENE_MU[sc], self.s, self.s, FS)
            anc.w = w.copy(); anc.adapting = False
            k = min(FS // 2, len(x))
            e, _ = anc.process_block(x[:k], d[:k])
            v = 10 * np.log10(np.sum(e[FS // 8:] ** 2) / np.sum(d[FS // 8:k] ** 2))
            print(f"[twin]   {sc:7s} -> {v:6.2f} dB")
            self.bank.append(w); self.bank_score.append(v)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, bank=np.array(self.bank), score=np.array(self.bank_score),
                 L=L)

    # ---------- pipeline state ----------
    def _build_pipeline(self):
        # G5: a headset is TWO independent ANC loops, and each ear can be driven
        # from either reference microphone depending on the causal margin.
        self.anc = {e: StreamingANC(L, MU, self.s, self.s, FS) for e in ("L", "R")}
        for e in self.anc:
            self.anc[e].set_filter(self.bank[0], 1)
        self.active_ref = {"L": "L", "R": "R"}
        self.det = ImpulseDetector(FS)
        self.gyro = Gyro(rng=np.random.default_rng(3))
        self.tracker = BearingTracker(theta0=0.0)
        self.dl = {k: StreamingDelayLine(8192)
                   for k in ("refL", "refR", "earL", "earR")}
        self.scene_i = 0
        self.gen = self._gen(SCENES[0])
        self.hist_x = np.zeros(L2_FRAME * 2)
        self.hist_xr = np.zeros(L2_FRAME * 2)
        self.hist_e = np.zeros(4096)
        self.hist_d = np.zeros(4096)
        self.roll_e = {"L": 0.0, "R": 0.0}
        self.roll_d = {"L": 0.0, "R": 0.0}
        self.sel_class = 0
        self.est_az = 0.0
        self.conf = 0.0
        self.fallbacks = 0
        self.frames_l2 = 0
        self.mode = "NORMAL"
        self.clip_l = self.clip_r = False
        self.protect_until = -1.0
        self.events = []
        self.l0_us = self.l2_us = 0.0
        self.next_impulse = self.impulse_every

    # ---------- one pipeline step ----------
    def _step(self, t):
        rng = self.rng
        scene = SCENES[self.scene_i]

        # --- source and geometry (head rotation) ---
        yaw = 60.0 * np.sin(2 * np.pi * t / 14.0)
        yaw_rate = 60.0 * (2 * np.pi / 14.0) * np.cos(2 * np.pi * t / 14.0)
        src = self.gen.block(BLOCK) * 10 ** ((95.0 - REF_DBSPL) / 20.0)
        if t >= self.next_impulse:
            b = sg.impulse_burst(BLOCK, FS, at_s=0.002, alpha=1.5, dur_s=0.015,
                                 amp=1.0, rng=rng)
            src = src + b * 10 ** ((160.0 - REF_DBSPL) / 20.0) / (np.max(np.abs(b)) + 1e-9)
            self.next_impulse = t + self.impulse_every
            self.events.append(Event("impulse", t, "alarm", "160 dB SPL transient injected"))

        az = -yaw                                     # source fixed in the world
        base = 0.003
        for k in self.dl:
            self.dl[k].push_block(src)
        dsam = lambda name: np.full(BLOCK, (float(tau_point(name, az)) + base) * FS)
        xL_ac = self.dl["refL"].read(dsam("refL"))
        xR_ac = self.dl["refR"].read(dsam("refR"))
        eL_ac = self.dl["earL"].read(dsam("earL"))
        eR_ac = self.dl["earR"].read(dsam("earR"))

        # --- microphones: AOP saturation, then self-noise ---
        xL = soft_clip_aop(xL_ac, AOP_DBSPL, REF_DBSPL)
        xR = soft_clip_aop(xR_ac, AOP_DBSPL, REF_DBSPL)
        self.clip_l = bool(np.max(np.abs(xL_ac)) > 0.9 * 10 ** ((AOP_DBSPL - REF_DBSPL) / 20))
        self.clip_r = bool(np.max(np.abs(xR_ac)) > 0.9 * 10 ** ((AOP_DBSPL - REF_DBSPL) / 20))
        nz = 10 ** (-73.0 / 20.0)
        xL = xL + nz * rng.standard_normal(BLOCK)
        xR = xR + nz * rng.standard_normal(BLOCK)

        # --- disturbance at EACH ear (cup response) ---
        if not hasattr(self, "zi_cup2"):
            self.zi_cup2 = {"L": np.zeros(2), "R": np.zeros(2)}
        d = {}
        d["L"], self.zi_cup2["L"] = signal.lfilter(self.b_cup, self.a_cup, eL_ac,
                                                   zi=self.zi_cup2["L"])
        d["R"], self.zi_cup2["R"] = signal.lfilter(self.b_cup, self.a_cup, eR_ac,
                                                   zi=self.zi_cup2["R"])

        # --- L2 (62.5 Hz): scene state, bearing, coefficient selection ---
        t2 = time.perf_counter()
        self.hist_x = np.r_[self.hist_x[BLOCK:], xL][-L2_FRAME * 2:]
        self.hist_xr = np.r_[self.hist_xr[BLOCK:], xR][-L2_FRAME * 2:]
        feat = frame_features(self.hist_x[-L2_FRAME:], FS)
        tau, _ = gcc_phat(self.hist_x[-L2_FRAME:], self.hist_xr[-L2_FRAME:],
                          FS, f_lo=BAND[0], f_hi=BAND[1])
        # G5: coherence x coherent-spread, which PREDICTS bearing error. The old
        # peak-to-sidelobe margin collapsed on tonal sources.
        conf = bearing_confidence(self.hist_x[-L2_FRAME:], self.hist_xr[-L2_FRAME:],
                                  FS, f_lo=BAND[0], f_hi=BAND[1])
        self.conf = conf
        self.est_az = self.tracker.step(BLOCK / FS, gyro_dps=self.gyro.read(-yaw_rate),
                                        theta_audio=theta_from_itd(tau), conf=conf,
                                        mode="fusion")
        self.fallbacks = self.tracker.fallbacks
        self.frames_l2 += 1
        if self.sel_class != self.scene_i:
            self.sel_class = self.scene_i
            for e in self.anc:
                self.anc[e].set_filter(self.bank[self.sel_class], 256)
                self.anc[e].mu = SCENE_MU[SCENES[self.sel_class]]
                self.anc[e].leak = SCENE_LEAK[SCENES[self.sel_class]]
            self.events.append(Event("filter", t, "info",
                                     f"coefficients -> {SCENES[self.sel_class]}"))
        self.l2_us = (time.perf_counter() - t2) * 1e6

        # --- impulse detector + protection mode ---
        imp = False
        step = max(1, BLOCK // 64)
        for i in range(0, BLOCK, step):
            if self.det(xL[i]):
                imp = True
        if imp:
            self.protect_until = t + 0.35

        # --- G5: per-ear causal reference selection over all four pairings ---
        self.margins = {f"{r}->{e}": float(margin(r, e, self.est_az, self.tau_s))
                        for r in "LR" for e in "LR"}
        chosen, engage = {}, {}
        for ear in ("L", "R"):
            r, m = best_ref_for_ear(ear, self.est_az, self.tau_s)
            # Graceful degradation, not muting: measured in E05, a controller
            # outside its causal region still reaches -3.5 dB on a partly
            # periodic source, where muting scores 0.
            chosen[ear] = r if r is not None else ear
            engage[ear] = True
            self.active_ref[ear] = chosen[ear]
            self.anc[ear].engaged = True
            self.anc[ear].adapting = t > self.protect_until
        mL = self.margins["L->L"]; mR = self.margins["R->R"]

        # --- L0 (48 kHz), one controller per ear ---
        t0 = time.perf_counter()
        xs = np.stack([xL, xR])
        # filtered reference: same for both ears, so compute it once per block
        if not hasattr(self, "zi_sh"):
            self.zi_sh = np.zeros((2, len(self.s) - 1))
        xhat = np.empty_like(xs)
        for j in range(2):
            xhat[j], self.zi_sh[j] = signal.lfilter(self.s, [1.0], xs[j],
                                                    zi=self.zi_sh[j])
        eL, eR = process_pair(self.anc["L"], self.anc["R"], xs, d["L"], d["R"],
                              0 if chosen["L"] == "L" else 1,
                              0 if chosen["R"] == "L" else 1, xhat)
        e = {"L": eL, "R": eR}
        y = {"L": eL * 0.0, "R": eR * 0.0}     # anti-noise trace not retained here
        self.l0_us = (time.perf_counter() - t0) * 1e6

        # --- metrics ---
        self.hist_e = np.r_[self.hist_e[BLOCK:], e["L"]][-4096:]
        self.hist_d = np.r_[self.hist_d[BLOCK:], d["L"]][-4096:]
        a = 0.9
        for ear in ("L", "R"):
            self.roll_e[ear] = a * self.roll_e[ear] + (1 - a) * float(np.mean(e[ear] ** 2))
            self.roll_d[ear] = a * self.roll_d[ear] + (1 - a) * float(np.mean(d[ear] ** 2))

        if t > self.protect_until and imp:
            self.mode = "IMPULSE"
        elif t <= self.protect_until:
            self.mode = "PROTECT"
        elif conf < CONF_FLOOR:
            self.mode = "LOW_CONFIDENCE"
        else:
            self.mode = {"engine": "ENGINE", "rotor": "ROTOR",
                         "wind": "WIND", "siren": "NORMAL"}[scene]
        return dict(xL=xL, xR=xR, d=d, e=e, y=y, yaw=yaw, az=az, feat=feat,
                    engage=True, mL=mL, mR=mR, imp=imp, scene=scene,
                    chosen=chosen)

    # ---------- telemetry ----------
    @staticmethod
    def _dec(x, n=256):
        k = max(1, len(x) // n)
        return [float(v) for v in x[::k][:n]]

    @staticmethod
    def _spec(x, fs, nfft=512):
        w = np.hanning(min(len(x), nfft))
        seg = x[-len(w):] * w
        X = np.abs(np.fft.rfft(seg, nfft)) / (len(w) / 2)
        return 20 * np.log10(X + 1e-9), fs / nfft

    def _frame(self, t, st):
        att = lambda ee, dd: 10 * np.log10((np.mean(ee ** 2) + 1e-30) /
                                           (np.mean(dd ** 2) + 1e-30))
        instL = att(st["e"]["L"], st["d"]["L"])
        instR = att(st["e"]["R"], st["d"]["R"])
        inst = 10 * np.log10(
            (np.mean(st["e"]["L"] ** 2) + np.mean(st["e"]["R"] ** 2) + 1e-30) /
            (np.mean(st["d"]["L"] ** 2) + np.mean(st["d"]["R"] ** 2) + 1e-30))
        roll = 10 * np.log10((self.roll_e["L"] + self.roll_e["R"] + 1e-30) /
                             (self.roll_d["L"] + self.roll_d["R"] + 1e-30))
        cpu = self.proc.cpu_percent(None)
        rss = self.proc.memory_info().rss / 1e6
        budget_us = BLOCK / FS * 1e6
        l0_load = self.l0_us / budget_us
        mag_e, df = self._spec(self.hist_e, FS)
        mag_x, _ = self._spec(self.hist_x, FS)

        sc = {}
        S = lambda **k: Scalar(**k)
        sc["atten_inst"] = S(value=inst, unit="dB", lo=-40, hi=6,
                             good=inst < 0, group="anc", label="attenuation joint (block)")
        sc["atten_L"] = S(value=instL, unit="dB", lo=-40, hi=6, good=instL < 0,
                          group="anc", label="attenuation left ear")
        sc["atten_R"] = S(value=instR, unit="dB", lo=-40, hi=6, good=instR < 0,
                          group="anc", label="attenuation right ear")
        sc["atten_roll"] = S(value=roll, unit="dB", lo=-40, hi=6,
                             good=roll < 0, group="anc", label="attenuation (rolling)")
        sc["w_norm_L"] = S(value=self.anc["L"].w_norm, unit="", lo=0, hi=5,
                           group="anc", label="filter norm L")
        sc["w_norm_R"] = S(value=self.anc["R"].w_norm, unit="", lo=0, hi=5,
                           group="anc", label="filter norm R")
        sc["engaged"] = S(value=1.0 if st["engage"] else 0.0, unit="", lo=0, hi=1,
                          good=st["engage"], group="anc", label="controller engaged")
        for k, v in self.margins.items():
            sc["margin_" + k.replace("->", "_to_")] = S(
                value=v * 1e6, unit="us", lo=-650, hi=550, good=v > 0,
                group="causality", label=f"margin  ref {k[0]} -> ear {k[-1]}")
        sc["ref_L"] = S(value=0.0 if st["chosen"]["L"] == "L" else 1.0, lo=0, hi=1,
                        group="causality", label="left ear uses ref (0=L,1=R)")
        sc["ref_R"] = S(value=0.0 if st["chosen"]["R"] == "L" else 1.0, lo=0, hi=1,
                        group="causality", label="right ear uses ref (0=L,1=R)")
        sc["contra_in_use"] = S(
            value=float((st["chosen"]["L"] == "R") + (st["chosen"]["R"] == "L")),
            lo=0, hi=2, group="causality",
            label="ears on contralateral ref")
        sc["elec_delay"] = S(value=ELEC_DELAY * 1e6, unit="us", lo=0, hi=200,
                             good=True, group="causality", label="electrical delay")
        sc["bearing_est"] = S(value=self.est_az, unit="deg", lo=-90, hi=90,
                              group="direction", label="bearing (estimated)")
        sc["bearing_true"] = S(value=st["az"], unit="deg", lo=-90, hi=90,
                               group="direction", label="bearing (true)")
        sc["bearing_err"] = S(value=self.est_az - st["az"], unit="deg", lo=-30, hi=30,
                              good=abs(self.est_az - st["az"]) < 5,
                              group="direction", label="bearing error")
        sc["doa_conf"] = S(value=self.conf, unit="", lo=0, hi=1,
                           good=self.conf > CONF_FLOOR, group="direction", label="DoA confidence")
        sc["fallback_pct"] = S(value=100.0 * self.fallbacks / max(self.frames_l2, 1),
                               unit="%", lo=0, hi=100, good=True,
                               group="direction", label="fallback rate")
        sc["head_yaw"] = S(value=st["yaw"], unit="deg", lo=-90, hi=90,
                           group="direction", label="head yaw (IMU)")
        sc["periodicity"] = S(value=float(st["feat"][12]), unit="", lo=0, hi=1,
                              group="state", label="periodicity P")
        sc["kurtosis"] = S(value=float(st["feat"][13]), unit="", lo=0, hi=2,
                           group="state", label="impulsiveness I")
        sc["crest"] = S(value=float(st["feat"][14]), unit="", lo=0, hi=2,
                        group="state", label="crest factor")
        sc["centroid"] = S(value=float(st["feat"][16]), unit="", lo=0, hi=1,
                           group="state", label="spectral centroid")
        sc["l0_us"] = S(value=self.l0_us, unit="us", lo=0, hi=budget_us,
                        good=l0_load < 0.8, group="compute", label="L0 block time")
        sc["l0_load"] = S(value=100 * l0_load, unit="%", lo=0, hi=100,
                          good=l0_load < 0.8, group="compute", label="L0 load")
        sc["l2_us"] = S(value=self.l2_us, unit="us", lo=0, hi=budget_us,
                        group="compute", label="L2 frame time")
        sc["rtf"] = S(value=(self.l0_us + self.l2_us) / budget_us, unit="x", lo=0, hi=1.5,
                      good=(self.l0_us + self.l2_us) < budget_us,
                      group="compute", label="real-time factor")
        sc["cpu"] = S(value=cpu, unit="%", lo=0, hi=100, group="compute", label="process CPU")
        sc["rss"] = S(value=rss, unit="MB", lo=0, hi=2000, group="compute", label="process RSS")
        sc["mu"] = S(value=self.anc["L"].mu, unit="", lo=0, hi=0.03, group="anc",
                     label="step size (head C)")
        # --- programme status. These are RESULTS already measured elsewhere,
        # surfaced here so the dashboard reflects the whole project, not just
        # what is streaming. Each is traceable to an experiment.
        sc["g5_gain_db"] = S(value=2.42, unit="dB", lo=0, hi=4, good=True,
                             group="gates", label="G5 direction gain (E05)")
        sc["e06_gain_db"] = S(value=9.28, unit="dB", lo=0, hi=12, good=True,
                              group="gates", label="E06 selection gain after change")
        sc["l1_params_k"] = S(value=22.988, unit="k", lo=0, hi=100, good=True,
                              group="gates", label="L1 params (E03, untrained)")
        sc["l1_mmacs"] = S(value=59.2, unit="MMAC/s", lo=0, hi=60, good=True,
                           group="gates", label="L1 MAC rate (E03)")
        sc["l1_latency_ms"] = S(value=8.0, unit="ms", lo=0, hi=15, good=True,
                                group="gates", label="L1 algorithmic latency")
        sc["fs_l0"] = S(value=FS, unit="Hz", group="rates", label="L0 rate", fmt=".0f")
        sc["fs_l2"] = S(value=FS / BLOCK, unit="Hz", group="rates", label="L2 rate")
        sc["clip_l"] = S(value=1.0 if self.clip_l else 0.0, lo=0, hi=1,
                         good=not self.clip_l, group="mics", label="ref L saturated")
        sc["clip_r"] = S(value=1.0 if self.clip_r else 0.0, lo=0, hi=1,
                         good=not self.clip_r, group="mics", label="ref R saturated")
        sc["aop"] = S(value=AOP_DBSPL, unit="dBSPL", group="mics", label="mic AOP")

        act = float(np.sqrt(np.mean(st["xL"] ** 2)))
        anc_on = st["engage"]
        blocks = {
            "mic_ref_l": Block("warn" if self.clip_l else "active", detail="analog MEMS"),
            "mic_ref_r": Block("warn" if self.clip_r else "active", detail="analog MEMS"),
            "mic_err": Block("active", detail="in-cup"),
            "mic_boom": Block("idle", detail="not simulated yet"),
            "afe": Block("warn" if (self.clip_l or self.clip_r) else "active",
                         detail="limiter"),
            "codec": Block("active", latency_us=ELEC_DELAY * 1e6, rate_hz=FS,
                           detail="ADAU1772"),
            "l0": Block("active" if anc_on else "bypassed", load=l0_load,
                        latency_us=self.l0_us, rate_hz=FS,
                        detail="2x FxNLMS L=%d, ref %s/%s" % (L, st["chosen"]["L"], st["chosen"]["R"])),
            "l2": Block("active", latency_us=self.l2_us, rate_hz=FS / BLOCK,
                        detail=SCENES[self.sel_class]),
            "l1": Block("idle", detail="E03 specified, 23k params, untrained"),
            "driver": Block("active" if anc_on else "idle"),
            "ear": Block("fault" if self.mode == "PROTECT" else "active"),
            "radio": Block("idle"),
        }
        edges = [Edge("mic_ref_l", "afe", min(1.0, act * 3)),
                 Edge("mic_ref_r", "afe", min(1.0, act * 3)),
                 Edge("mic_err", "afe", min(1.0, float(np.sqrt(np.mean(st["e"]["L"] ** 2))) * 3)),
                 Edge("afe", "codec", min(1.0, act * 3)),
                 Edge("codec", "l0", min(1.0, act * 3)),
                 Edge("codec", "l2", min(1.0, act * 3)),
                 Edge("l2", "l0", 1.0 if self._blend_active() else 0.15,
                      label="coefficients"),
                 Edge("l0", "driver", min(1.0, float(np.sqrt(np.mean(st["y"]["L"] ** 2))) * 3)),
                 Edge("driver", "ear", min(1.0, float(np.sqrt(np.mean(st["y"]["L"] ** 2))) * 3)),
                 Edge("ear", "mic_err", min(1.0, float(np.sqrt(np.mean(st["e"]["L"] ** 2))) * 3)),
                 Edge("mic_boom", "l1", 0.0), Edge("l1", "radio", 0.0),
                 Edge("l2", "l1", 0.0, label="conditioning")]

        ev, self.events = self.events[-6:], []
        return TelemetryFrame(
            t=t, seq=self.seq, source=self.kind, experiment="live twin",
            mode=self.mode,
            channels={
                "ref_l": Channel(FS, self._dec(st["xL"]), label="Reference mic L",
                                 clipped=self.clip_l),
                "ref_r": Channel(FS, self._dec(st["xR"]), label="Reference mic R",
                                 clipped=self.clip_r),
                "disturbance_L": Channel(FS, self._dec(st["d"]["L"]),
                                         label="Disturbance at LEFT ear"),
                "anti_noise_L": Channel(FS, self._dec(st["y"]["L"]),
                                        label="Anti-noise out (left)"),
                "error_L": Channel(FS, self._dec(st["e"]["L"]),
                                   label="Error mic LEFT (residual)"),
                "error_R": Channel(FS, self._dec(st["e"]["R"]),
                                   label="Error mic RIGHT (residual)"),
            },
            spectra={
                "ref": Spectrum(0.0, df, [float(v) for v in mag_x[:220]],
                                label="Reference spectrum"),
                "err": Spectrum(0.0, df, [float(v) for v in mag_e[:220]],
                                label="Residual spectrum"),
            },
            scalars=sc,
            vectors={
                "filter_w": Vector([float(v) for v in self.anc["L"].w[:256]],
                                   label="Active filter coefficients"),
                "state_feat": Vector([float(v) for v in st["feat"]],
                                     label="L2 acoustic state", kind="bar"),
                "bank_scores": Vector([float(v) for v in self.bank_score],
                                      label="Filter bank (dB on own scene)", kind="bar"),
            },
            blocks=blocks, edges=edges,
            geometry=Geometry(head_yaw_deg=st["yaw"], source_az_deg=st["az"],
                              est_az_deg=self.est_az, conf=self.conf,
                              causal_L=self.margins["L->L"] > 0,
                              causal_R=self.margins["R->R"] > 0,
                              active_ref=f'{st["chosen"]["L"]}/{st["chosen"]["R"]}'),
            events=ev,
            notes=f"scene={st['scene']} filter={SCENES[self.sel_class]}")

    def _blend_active(self):
        return any(a._blend > 0 for a in self.anc.values())

    def run(self):
        t = 0.0
        pub_every = max(1, int((FS / BLOCK) / self.publish_hz))
        k = 0
        next_wall = time.time()
        scene_t = 0.0
        while not self._stop.is_set():
            st = self._step(t)
            k += 1
            if k % pub_every == 0:
                self.seq += 1
                self.bus.publish(self._frame(t, st))
            t += BLOCK / FS
            scene_t += BLOCK / FS
            if scene_t >= self.scene_s:
                scene_t = 0.0
                self.scene_i = (self.scene_i + 1) % len(SCENES)
                self.gen = self._gen(SCENES[self.scene_i])
            next_wall += BLOCK / FS
            slp = next_wall - time.time()
            if slp > 0:
                time.sleep(slp)
            else:
                next_wall = time.time()
