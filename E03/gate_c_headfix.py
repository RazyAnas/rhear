#!/usr/bin/env python3
"""Gate C failed slowly, not flatly. This tests why, and whether it is fixable.

The result: phase alignment climbed 0.8384 -> 0.8662 over 1500 steps against an
identity baseline of 0.8510. It DID beat the identity, monotonically, and was
still rising at the cut -- but it took ~190 steps just to get back to identity
and gained only +0.015 in total, on a 16-clip MEMORISATION task with every other
weight frozen. That is far too slow to trust.

Hypothesis: the bottleneck, not the idea. The head reads the DPRNN output at
frequency width 6 and upsamples ~30x to 183 bins. Phase varies fast across
frequency -- much faster than magnitude -- so 6 numbers cannot carry it. The
main decoder does not have this problem because it has SKIP CONNECTIONS from the
encoder at every width; the phase head has none.

Test: the same head, given skips. The full-band branch already computes
bin-domain features on the way down (257 -> 65 -> 17 -> 6), so its intermediate
activations are exactly the right thing to skip into a bin-domain phase head.

If V2 converges where V1 crawled, the plan survives with a wider head. If both
crawl, the phase target is not reachable from this trunk and the plan needs a
different structure -- which is worth knowing before anyone spends a GPU day.

    /opt/anaconda3/bin/python E03/gate_c_headfix.py
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import N_FFT, FS, CausalDeconvBlock          # noqa: E402
from gtcrn_phase import GTCRNPhase, PhaseHead, PHASE_BINS    # noqa: E402
from train_interim import Pairs, stft, DEV                   # noqa: E402
from rhear_data.manifest import read_manifest                # noqa: E402
from gates_bc import load_g7_into                            # noqa: E402


class PhaseHeadSkip(nn.Module):
    """Per-bin phase with skips from the full-band branch's own bin-domain
    activations. Same output contract as PhaseHead: unit (cos, sin) per bin,
    residual around identity, no atan2."""

    def __init__(self, c3, fb_ch=(8, 12, 16)):
        super().__init__()
        f0, f1, f2 = fb_ch                      # activations at 65, 17, 6 bins
        self.d0 = CausalDeconvBlock(c3 + f2, 24, stride=(1, 3))    # 6  -> 17
        self.d1 = CausalDeconvBlock(24 + f1, 16, stride=(1, 4))    # 17 -> 65
        self.d2 = CausalDeconvBlock(16 + f0, 2, stride=(1, 3), last=True)  # 65 -> 183

    def forward(self, x, fb_acts):
        a65, a17, a6 = fb_acts
        x = self.d0(torch.cat([x, a6], 1), 17)
        x = self.d1(torch.cat([x, a17], 1), 65)
        x = self.d2(torch.cat([x, a65], 1), PHASE_BINS)
        cos = x[:, :1] + 1.0
        sin = x[:, 1:]
        n = torch.sqrt(cos ** 2 + sin ** 2 + 1e-9)
        return cos / n, sin / n


def trunk(model, spec, want_fb=False):
    """Encoder + full-band + fuse + DPRNN, returning the phase head's input and,
    optionally, the full-band branch's intermediate activations."""
    mag = torch.sqrt(spec[:, :1] ** 2 + spec[:, 1:] ** 2 + 1e-9)
    x0 = torch.cat([spec, mag], 1)
    x = model.erb(x0)
    for e in model.enc:
        x = e(x)
    z = x0
    acts = []
    for b in model.fb:
        z = b(z)
        acts.append(z)
    x = model.fuse(torch.cat([x, z], 1))
    x = model.dprnn(x)
    return (x, acts) if want_fb else x


def run(head, model, Xb, tgt_cos, tgt_sin, w, steps, lr, label):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    spec = torch.stack([Xb.real, Xb.imag], 1).transpose(2, 3)
    hist = []
    with torch.no_grad():
        base_align = float((w * tgt_cos).sum() / w.sum())
    for s in range(steps):
        opt.zero_grad()
        with torch.no_grad():
            h, acts = trunk(model, spec, want_fb=True)
        cos, sin = (head(h, acts) if isinstance(head, PhaseHeadSkip) else head(h))
        l = (w * ((cos - tgt_cos) ** 2 + (sin - tgt_sin) ** 2)).mean()
        l.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        if s % max(1, steps // 6) == 0 or s == steps - 1:
            with torch.no_grad():
                align = float((w * (cos * tgt_cos + sin * tgt_sin)).sum() / w.sum())
            hist.append({"step": s, "loss": float(l), "align": align})
            print(f"    {label:14s} step {s:5d}   loss {float(l):8.5f}   "
                  f"align {align:+.4f}  ({align - base_align:+.4f} vs identity)")
    return hist, base_align


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
    ap.add_argument("--train-data", default=os.path.join(HERE, "..", "handoff", "data", "h3_20k"))
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "gate_c_headfix.json"))
    a = ap.parse_args()

    torch.manual_seed(0)
    win = torch.hann_window(N_FFT, device=DEV)

    model = GTCRNPhase().to(DEV)
    load_g7_into(model, a.ckpt)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hdr, rows = read_manifest(os.path.join(a.train_data, "manifest.jsonl"))
    ds = Pairs(a.train_data, "train", rows, cache=False)
    xb = torch.stack([ds[i][0] for i in range(a.clips)]).to(DEV)
    yb = torch.stack([ds[i][1] for i in range(a.clips)]).to(DEV)

    hop = int(model.hop_.item())
    with torch.no_grad():
        Xb = stft(xb, win, hop)
        Sb = stft(yb, win, hop)
        d = (Sb / (Sb.abs() + 1e-9)) * torch.conj(Xb / (Xb.abs() + 1e-9))
        tgt_cos = d.real.transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        tgt_sin = d.imag.transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        w = Sb.abs().transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        w = w / (w.mean() + 1e-9)

    print(f"\nclips     {a.clips}   steps {a.steps}   lr {a.lr}")
    print(f"target    per-bin rotation from noisy phase to clean phase, "
          f"magnitude-weighted\n")

    v1 = PhaseHead(64).to(DEV)
    v2 = PhaseHeadSkip(64).to(DEV)
    print(f"  V1 (no skips)   {sum(p.numel() for p in v1.parameters()):,} params")
    print(f"  V2 (fb skips)   {sum(p.numel() for p in v2.parameters()):,} params\n")

    h1, base = run(v1, model, Xb, tgt_cos, tgt_sin, w, a.steps, a.lr, "V1 no skips")
    print()
    h2, _ = run(v2, model, Xb, tgt_cos, tgt_sin, w, a.steps, a.lr, "V2 fb skips")

    g1 = h1[-1]["align"] - base
    g2 = h2[-1]["align"] - base
    print(f"\n  identity alignment              {base:+.4f}")
    print(f"  V1 gain over identity           {g1:+.4f}")
    print(f"  V2 gain over identity           {g2:+.4f}")
    print(f"  V2 / V1                         {g2 / g1 if g1 else float('nan'):.2f}x")

    verdict = ("V2 is decisively better — the bottleneck was the cause, and the "
               "plan\n  survives with a skip-connected phase head."
               if g2 > 1.5 * g1 else
               "V2 is not decisively better — the limit is not the bottleneck. "
               "The phase\n  target is hard to reach from this trunk; do not "
               "spend a GPU day yet.")
    print(f"\n  {verdict}")

    with open(a.out, "w") as f:
        json.dump({"identity_alignment": base, "v1": h1, "v2": h2,
                   "gain_v1": g1, "gain_v2": g2}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
