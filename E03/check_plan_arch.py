#!/usr/bin/env python3
"""Feasibility check for the chosen plan's architecture, before anyone builds it.

The plan is 96 ERB bands + a per-bin phase head below 4 kHz. Two things in
gtcrn_lite.py block it today, and this checks that both have a cheap fix and
that the result still satisfies the properties the project depends on.

BLOCKER 1 -- the full-band branch is hardwired to the 48-band encoder width.
    At 48 bands the encoder's strides (2,2,2,1) give 48 -> 24 -> 12 -> 6, and
    the full-band branch's (4,4,3) lands 257 on exactly 6. At 96 bands the
    encoder gives 12 and the assert at gtcrn_lite.py:235 fires.

    Fix, and it is the cheap one: take the encoder's LAST stride from 1 to 2, so
    96 -> 48 -> 24 -> 12 -> 6. The encoder width is 6 again, the full-band
    branch is untouched, the DPRNN sees exactly the shape it sees today, and
    the decoder mirrors it with strides (2,2,2,2) instead of (1,2,2,2). No new
    operator, no resize, no padding -- the constraints that drove the original
    stride choices are all still met.

BLOCKER 2 -- there is no per-bin phase head.
    The deep-filter head already has exactly the right geometry: a deconv chain
    from the encoder width up to 183 bins (5.7 kHz), emitting PER BIN. 4 kHz is
    bin 128, inside 183. So the phase head is that same chain with 2 output
    channels (cos, sin) instead of 2*df_taps.

    This is also why it should not repeat the old phase branch's failure. That
    branch emitted one rotation per ERB band and spread it across bins up to 19
    wide (gtcrn_lite.py:124). This emits one per BIN.

Checks run:
    1. the 96-band variant builds and forwards
    2. parameters and MAC/s against the PS caps and the ESP32-S3 core
    3. gradients flow to every parameter, including the new head
    4. CAUSALITY: frame t's output does not move when future frames change
    5. the phase head starts at identity, so fine-tuning cannot destroy stage 1

    /opt/anaconda3/bin/python E03/check_plan_arch.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import (GTCRNLite, CausalConvBlock, CausalDeconvBlock,  # noqa: E402
                        N_BINS, FS)
from mac_by_bands import count_macs                                     # noqa: E402

HOP = 256
FRAME_RATE = FS / HOP
CORE_MMACS = 200.0
PS_MMACS = 125.0
PHASE_BINS = 183                      # what strides (4,4,2) k=3 give from 6
CUT_HZ = 4000

ok = []


def check(name, passed, detail=""):
    ok.append(bool(passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


class PhaseHead(nn.Module):
    """Per-bin phase, emitted as (cos, sin) so there is no atan2 in the graph.

    Residual around the identity, the same trick the deep-filter head uses: the
    head outputs a DELTA and 1 is added to the cosine channel, so a freshly
    initialised head is the identity rotation and fine-tuning from a magnitude-
    only checkpoint starts exactly where that checkpoint left off instead of
    scrambling it. atan2 is also the operator CMSIS-NN cannot run, which is one
    of the reasons the original phase branch was removed.
    """

    def __init__(self, c3):
        super().__init__()
        self.dec = nn.ModuleList([
            CausalDeconvBlock(c3, 24, stride=(1, 4)),
            CausalDeconvBlock(24, 16, stride=(1, 4)),
            CausalDeconvBlock(16, 2, stride=(1, 2), last=True),
        ])

    def forward(self, x):
        for i, d in enumerate(self.dec):
            x = d(x, (23, 91, PHASE_BINS)[i])
        cos = x[:, :1] + 1.0
        sin = x[:, 1:]
        n = torch.sqrt(cos ** 2 + sin ** 2 + 1e-9)
        return cos / n, sin / n


def build_96band(ch=(32, 48, 48, 64)):
    """The plan's variant: 96 bands, encoder re-strided so the width stays 6."""
    m = GTCRNLite(ch=ch, n_bands=96, phase=False, fullband=True, df=False, hop=HOP)
    c0, c1, c2, c3 = ch
    m.enc = nn.ModuleList([
        CausalConvBlock(3, c0, stride=(1, 2), groups=1),
        CausalConvBlock(c0, c1, stride=(1, 2)),
        CausalConvBlock(c1, c2, stride=(1, 2)),
        CausalConvBlock(c2, c3, stride=(1, 2)),        # was (1,1)
    ])
    m.dec = nn.ModuleList([
        CausalDeconvBlock(c3 * 2, c2, stride=(1, 2)),  # was (1,1)
        CausalDeconvBlock(c2 * 2, c1, stride=(1, 2)),
        CausalDeconvBlock(c1 * 2, c0, stride=(1, 2)),
        CausalDeconvBlock(c0 * 2, 1, stride=(1, 2), last=True),
    ])
    m.phase_head = PhaseHead(c3)
    return m


def main() -> None:
    torch.manual_seed(0)
    df_hz = FS / 512
    cut_bin = int(round(CUT_HZ / df_hz))
    print(f"\nplan      96 ERB bands + per-bin phase below {CUT_HZ/1000:.0f} kHz")
    print(f"          {CUT_HZ/1000:.0f} kHz = bin {cut_bin} of {N_BINS}; "
          f"the head covers {PHASE_BINS} bins "
          f"({PHASE_BINS * df_hz / 1000:.1f} kHz)\n")

    print("1. builds and forwards")
    m = build_96band()
    m.eval()
    spec = torch.randn(2, 2, 40, N_BINS)
    try:
        with torch.no_grad():
            mm, mp, spp, dfc = m(spec)
            cos, sin = m.phase_head(m.dprnn(_encode(m, spec)))
        check("96-band forward", mm.shape[-1] == N_BINS, f"mask {tuple(mm.shape)}")
        check("phase head forward", cos.shape[-1] == PHASE_BINS,
              f"cos/sin {tuple(cos.shape)}")
        check("phase head covers the cutoff", PHASE_BINS >= cut_bin,
              f"{PHASE_BINS} bins >= bin {cut_bin}")
    except Exception as e:                                       # noqa: BLE001
        check("96-band forward", False, f"{type(e).__name__}: {e}")
        _summary()
        return

    print("\n2. cost against the budgets")
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    base = GTCRNLite(ch=(32, 48, 48, 64), n_bands=48, phase=False,
                     fullband=True, df=False, hop=HOP)
    base_params = sum(p.numel() for p in base.parameters() if p.requires_grad)
    mac_base = count_macs(base)
    mac_new = count_macs(m) + _phase_macs(m, spec)
    # count_macs omits STFT/ISTFT and counts transposed convs by input, where
    # the preflight measured 32.34 MMAC/s for the base. Scale by that ratio so
    # the comparison is against the same accounting the budgets are quoted in.
    scale = 32.34 / (mac_base * FRAME_RATE / 1e6)
    mmacs = mac_new * FRAME_RATE / 1e6 * scale
    print(f"  params      {params:,}  (base {base_params:,})   "
          f"~{params / 1024:.0f} KB int8")
    print(f"  compute     {mmacs:.1f} MMAC/s  (base 32.3 measured)   "
          f"{100 * mmacs / CORE_MMACS:.0f}% of one core")
    check("model under the PS 200 KB cap", params / 1024 < 200,
          f"{params / 1024:.0f} KB int8")
    check("compute under the PS 125 MMAC/s cap", mmacs < PS_MMACS,
          f"{mmacs:.1f} MMAC/s")
    check("compute fits one ESP32-S3 core", mmacs < CORE_MMACS,
          f"{100 * mmacs / CORE_MMACS:.0f}% of ~200 MMAC/s")

    print("\n3. gradients reach every parameter")
    m.train()
    mm, mp, spp, dfc = m(spec)
    cos, sin = m.phase_head(m.dprnn(_encode(m, spec)))
    (mm.mean() + cos.mean() + sin.mean() + spp.mean()).backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()
                                    or p.grad.abs().sum() == 0)]
    check("no parameter without a finite, non-zero gradient", not dead,
          "all reached" if not dead else f"{len(dead)} dead: {dead[:4]}")
    head_grads = [n for n, p in m.phase_head.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0]
    check("phase head trains", len(head_grads) > 0,
          f"{len(head_grads)} tensors with gradient")

    print("\n4. causality — future frames must not move the past")
    m.eval()
    with torch.no_grad():
        a = torch.randn(1, 2, 32, N_BINS)
        b = a.clone()
        b[:, :, 20:] = torch.randn_like(b[:, :, 20:])       # rewrite the future
        ma, _, _, _ = m(a)
        mb, _, _, _ = m(b)
        drift = (ma[:, :, :20] - mb[:, :, :20]).abs().max().item()
        ca, sa = m.phase_head(m.dprnn(_encode(m, a)))
        cb, sb = m.phase_head(m.dprnn(_encode(m, b)))
        pdrift = max((ca[:, :, :20] - cb[:, :, :20]).abs().max().item(),
                     (sa[:, :, :20] - sb[:, :, :20]).abs().max().item())
    check("mask is causal", drift < 1e-5, f"max drift {drift:.2e}")
    check("phase head is causal", pdrift < 1e-5, f"max drift {pdrift:.2e}")

    print("\n5. the phase head starts at identity")
    fresh = PhaseHead(64).eval()
    with torch.no_grad():
        c, s = fresh(torch.randn(1, 64, 12, 6))
        ang = torch.atan2(s, c).abs().max().item()
    check("fresh head is within 0.2 rad of no rotation", ang < 0.2,
          f"max |rotation| {ang:.3f} rad")
    print("     (so it can be fine-tuned onto G7-base rather than retrained)")

    _summary()


def _encode(m, spec):
    """Run the encoder + fuse the way forward() does, to feed the phase head."""
    mag = torch.sqrt(spec[:, :1] ** 2 + spec[:, 1:] ** 2 + 1e-9)
    x = torch.cat([spec, mag], 1)
    z_in = x
    x = m.erb(x)
    for e in m.enc:
        x = e(x)
    if m.fullband:
        z = z_in
        for b in m.fb:
            z = b(z)
        x = m.fuse(torch.cat([x, z], 1))
    return x


def _phase_macs(m, spec):
    from mac_by_bands import count_macs as _c

    class Wrap(nn.Module):
        def __init__(self, head, enc):
            super().__init__()
            self.head = head
            self.enc = enc

        def forward(self, s):
            return self.head(self.enc[0](self.enc[1](s)))

    # Counting the head alone is enough: the encoder is already in count_macs(m).
    total = {"n": 0}

    def hook(mod, inp, out):
        x = inp[0]
        per_in = (mod.out_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        total["n"] += x.numel() * per_in

    hs = [mm.register_forward_hook(hook) for mm in m.phase_head.modules()
          if isinstance(mm, nn.ConvTranspose2d)]
    frames = 40
    with torch.no_grad():
        m.phase_head(torch.randn(1, 64, frames, 6))
    for h in hs:
        h.remove()
    return total["n"] / frames


def _summary():
    n, t = sum(ok), len(ok)
    print(f"\n{'=' * 60}")
    print(f"{n}/{t} checks passed")
    if n == t:
        print("The plan's architecture is buildable, fits both budgets, trains,\n"
              "stays causal, and starts as a no-op on top of G7-base.")
    else:
        print("At least one check failed. Do not hand this to anyone to build\n"
              "until the failure is understood.")
    print("=" * 60)


if __name__ == "__main__":
    main()
