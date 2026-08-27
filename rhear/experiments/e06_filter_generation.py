"""E06 - Does neural filter selection actually beat the alternatives?

This is the project's central claim, tested. On non-stationary noise, compare:

  1. fixed generic     one broadband filter, never changes, no adaptation
  2. FxNLMS only       adapts continuously from zero -- the classical baseline
  3. oracle + refine   perfect knowledge of the noise class selects a pre-trained
                       filter, FxNLMS refines it  -> the ceiling for selection
  4. learned + refine  a small trained network does the selecting  -> what we can
                       actually build

Reported over the whole scenario and, separately, over the 250 ms following each
noise change -- which is where selection is supposed to earn its place.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from rhear.sim import paths
from rhear.core import signals as sg
from rhear.core.state import frame_features
from rhear.core.l2runtime import run_controller, train_filter, sliding_nmse_db

FS = 48_000
FRAME = 768                      # 16 ms at 48 kHz -> 62.5 Hz, the L2 rate
# E01/F4 design rule: to exploit periodicity the filter must span at least one
# fundamental period. The lowest fundamental here is the 50 Hz engine -> 960
# samples at 48 kHz. L=256 silently failed to train the engine filter on the
# first run of this experiment, exactly as that rule predicts.
L = 1024
ELEC_DELAY = 38e-6               # ADAU1772, the part we selected
MU = 0.01
MIC_SNR_DB = 45.0                # analog MEMS self-noise; caps attenuation at the
                                 # coherence bound, per E00/V2
CLASSES = ["engine", "rotor", "wind", "siren"]


def measure(x_ac, rng):
    """What the reference mic delivers: the acoustic field plus its own noise.
    The disturbance d is driven by the clean field, so reference and disturbance
    are imperfectly coherent -- which is what caps achievable attenuation."""
    return x_ac + 10 ** (-MIC_SNR_DB / 20.0) * rng.standard_normal(len(x_ac))


def make_noise(kind, n, rng):
    if kind == "engine":
        return sg.harmonic(n, FS, f0=50.0, n_harm=10, rng=rng)
    if kind == "rotor":
        return sg.harmonic(n, FS, f0=120.0, n_harm=6, rng=rng)
    if kind == "wind":
        return sg.broadband(n, FS, 60, 900, rng=rng)
    if kind == "siren":
        t = np.arange(n) / FS
        f = 700 + 400 * signal.sawtooth(2 * np.pi * 0.7 * t, width=0.5)
        return np.sin(2 * np.pi * np.cumsum(f) / FS).astype(float)
    raise ValueError(kind)


class Selector(nn.Module):
    """L2 head A + head B, minimal: features -> class posterior -> filter choice."""

    def __init__(self, d_in, n_cls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU(),
                                 nn.Linear(32, n_cls))

    def forward(self, x):
        return self.net(x)


def train_selector(rng, n_per_class=400):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(n_per_class):
            seg = make_noise(c, FRAME, rng) * rng.uniform(0.3, 3.0)
            seg = seg + 0.02 * rng.standard_normal(FRAME)       # mic self-noise
            X.append(frame_features(seg, FS)); y.append(ci)
    X = torch.tensor(np.array(X)); y = torch.tensor(y)
    mu_, sd_ = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu_) / sd_
    net = Selector(X.shape[1], len(CLASSES))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(300):
        opt.zero_grad(); out = net(Xn); l = lossf(out, y); l.backward(); opt.step()
    acc = (net(Xn).argmax(1) == y).float().mean().item()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  selector: {n_par} params, train accuracy {acc*100:.1f}%")
    return net, mu_, sd_, n_par


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    t0 = time.time()
    p = paths.primary_path(FS, ntaps=512)
    s_ac = paths.secondary_path(FS, ntaps=1024)
    s = paths.add_electrical_delay(s_ac, ELEC_DELAY, FS)

    # ---- pre-trained filter bank (the SFANC/GFANC premise) ----
    print("Training the fixed-filter bank (one filter per noise class):")
    bank = {}
    for c in CLASSES:
        xac = make_noise(c, int(FS * 1.5), rng)
        xt, dt = measure(xac, rng), signal.lfilter(p, [1.0], xac)
        bank[c] = train_filter(xt, dt, s, L=L, mu=MU, passes=3)
        r = run_controller(xt, dt, s, s, L=L, w_init=bank[c], refine=False)
        v = 10 * np.log10(np.nansum(r["e"][FS // 2:] ** 2) / np.sum(dt[FS // 2:] ** 2))
        print(f"    {c:8s} -> {v:6.2f} dB on its own noise")
    xgac = sum(make_noise(c, int(FS * 1.5), rng) for c in CLASSES) / 2.0
    xg, dg = measure(xgac, rng), signal.lfilter(p, [1.0], xgac)
    bank["generic"] = train_filter(xg, dg, s, L=L, mu=MU, passes=3)
    print("    generic  -> trained on the mixture\n")

    # ---- the learned selector ----
    print("Training L2 selector:")
    net, mu_, sd_, n_par = train_selector(rng)
    print()

    # ---- non-stationary scenario ----
    order = ["engine", "rotor", "wind", "engine", "siren", "wind"]
    seg_n = int(FS * 1.0)
    x_ac = np.concatenate([make_noise(c, seg_n, rng) for c in order])
    truth = np.repeat(np.array([CLASSES.index(c) for c in order]), seg_n)
    x, d = measure(x_ac, rng), signal.lfilter(p, [1.0], x_ac)
    n = len(x)
    switch_frames = [i * seg_n // FRAME for i in range(1, len(order))]
    print(f"Scenario: {' -> '.join(order)}  ({n/FS:.0f} s, {len(switch_frames)} changes)")
    print(f"Reference-mic self-noise {MIC_SNR_DB:.0f} dB SNR -> coherence bound "
          f"caps attenuation near {-MIC_SNR_DB:.0f} dB\n")

    def oracle_sel(state={"last": None}):
        def f(fi, hist):
            i = min(fi * FRAME, n - 1)
            c = CLASSES[truth[i]]
            if c != state["last"]:
                state["last"] = c
                return bank[c]
            return None
        return f

    def learned_sel(state={"last": None, "correct": 0, "total": 0}):
        def f(fi, hist):
            if len(hist) < FRAME:
                return None
            feat = frame_features(hist[-FRAME:], FS)
            with torch.no_grad():
                z = (torch.tensor(feat) - mu_) / sd_
                ci = int(net(z.unsqueeze(0)).argmax(1).item())
            i = min(fi * FRAME, n - 1)
            state["total"] += 1
            state["correct"] += int(ci == truth[i])
            c = CLASSES[ci]
            if c != state["last"]:
                state["last"] = c
                return bank[c]
            return None
        return f
    lsel_state = {"last": None, "correct": 0, "total": 0}

    runs = {}
    runs["fixed generic"] = run_controller(x, d, s, s, L=L, w_init=bank["generic"],
                                           refine=False, frame=FRAME)
    runs["FxNLMS only"] = run_controller(x, d, s, s, L=L, mu=MU, refine=True, frame=FRAME)
    runs["oracle + refine"] = run_controller(x, d, s, s, L=L, mu=MU, refine=True,
                                             frame=FRAME, selector=oracle_sel())

    def lf(fi, hist):
        if len(hist) < FRAME:
            return None
        feat = frame_features(hist[-FRAME:], FS)
        with torch.no_grad():
            z = (torch.tensor(feat) - mu_) / sd_
            ci = int(net(z.unsqueeze(0)).argmax(1).item())
        i = min(fi * FRAME, n - 1)
        lsel_state["total"] += 1
        lsel_state["correct"] += int(ci == truth[i])
        c = CLASSES[ci]
        if c != lsel_state["last"]:
            lsel_state["last"] = c
            return bank[c]
        return None
    runs["learned + refine"] = run_controller(x, d, s, s, L=L, mu=MU, refine=True,
                                              frame=FRAME, selector=lf)

    # ---- scoring ----
    win_s = 0.05
    print(f"{'controller':<20}{'overall':>12}{'after a change':>18}{'steady state':>16}")
    print("-" * 66)
    res = {}
    for name, r in runs.items():
        e = r["e"]
        ok = np.all(np.isfinite(e))
        overall = 10 * np.log10(np.nansum(e ** 2) / np.sum(d ** 2)) if ok else np.nan
        sl = sliding_nmse_db(e, d, FS, win_s)
        post, steady = [], []
        for k in range(1, len(order)):
            a = int(k * seg_n / FS / win_s)
            post += list(sl[a:a + int(0.25 / win_s)])          # 250 ms after a change
            steady += list(sl[a + int(0.5 / win_s):a + int(1.0 / win_s)])
        res[name] = (overall, np.nanmean(post), np.nanmean(steady))
        print(f"{name:<20}{overall:9.2f} dB{np.nanmean(post):15.2f} dB{np.nanmean(steady):13.2f} dB")
    print("-" * 66)
    acc = lsel_state["correct"] / max(lsel_state["total"], 1)
    print(f"learned selector frame accuracy on the scenario: {acc*100:.1f}%")
    o, l = res["oracle + refine"][1], res["learned + refine"][1]
    b = res["FxNLMS only"][1]
    print(f"after a change: FxNLMS {b:.2f} dB | learned {l:.2f} dB | oracle {o:.2f} dB")
    if abs(o - b) > 1e-6:
        print(f"learned selection closes {100*(l-b)/(o-b):.0f}% of the gap "
              f"between plain FxNLMS and the oracle")
    print(f"\n({time.time()-t0:.0f} s)")
    np.savez("results/e06_filter_generation.npz",
             **{k.replace(" ", "_").replace("+", "p"): np.array(v) for k, v in res.items()},
             selector_params=n_par, selector_acc=acc)
    print("saved -> results/e06_filter_generation.npz")
