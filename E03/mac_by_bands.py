#!/usr/bin/env python3
"""MAC/s for each band-count variant, so a plan can be checked against the board.

Parameter count does not depend on the band count -- the ERB matrix is fixed and
the convolutions are channel-wise, so every variant has the same 49,663 weights.
What changes is the length of the frequency axis every layer walks, and that is
pure compute. This counts it with forward hooks rather than modelling it, so the
answer holds for whatever the architecture actually is.

Budgets it is checked against:
    ESP32-S3   ~200 MMAC/s per core (0.834 MAC/cycle int16 at 240 MHz)
    PS 26052   <= 125 MMAC/s, model < 200 KB, RTF <= 0.75

    /opt/anaconda3/bin/python E03/mac_by_bands.py
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

from gtcrn_lite import GTCRNLite, N_BINS      # noqa: E402

FS = 16000
HOP = 256
FRAME_RATE = FS / HOP                          # 62.5 fps
CORE_MMACS = 200.0
PS_MMACS = 125.0


def count_macs(model, frames=64):
    """MACs for one forward pass over `frames` frames, via hooks."""
    total = {"n": 0}

    def conv_hook(m, inp, out):
        # out: (B,C,H,W). Each output element costs in_ch/groups * kh * kw MACs.
        per_out = (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        total["n"] += out.numel() * per_out

    def deconv_hook(m, inp, out):
        # A transposed conv costs by its INPUT elements, not its output.
        x = inp[0]
        per_in = (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        total["n"] += x.numel() * per_in

    def lin_hook(m, inp, out):
        total["n"] += out.numel() * m.in_features

    def gru_hook(m, inp, out):
        x = inp[0]
        steps = x.shape[1] if m.batch_first else x.shape[0]
        seqs = x.numel() // (steps * m.input_size)
        # 3 gates, each an (input + hidden) -> hidden matmul, per step.
        per_step = 3 * (m.input_size + m.hidden_size) * m.hidden_size
        total["n"] += seqs * steps * per_step * m.num_layers

    hs = []
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            hs.append(m.register_forward_hook(deconv_hook))
        elif isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.GRU):
            hs.append(m.register_forward_hook(gru_hook))

    spec = torch.randn(1, 2, frames, N_BINS)
    with torch.no_grad():
        model(spec)
    for h in hs:
        h.remove()
    return total["n"] / frames


def main() -> None:
    print(f"\nframe rate {FRAME_RATE:.1f} fps (hop {HOP} at {FS} Hz)")
    print(f"budgets    ESP32-S3 core ~{CORE_MMACS:.0f} MMAC/s   "
          f"PS cap {PS_MMACS:.0f} MMAC/s\n")

    print(f"{'variant':28s}{'params':>10s}{'MAC/frame':>12s}{'MMAC/s':>10s}"
          f"{'% core':>9s}   verdict")
    print("-" * 88)

    notes = []
    for nb in (48, 64, 96, 128, 256):
        for fb in (True, False):
            try:
                m = GTCRNLite(ch=(32, 48, 48, 64), n_bands=nb, phase=False,
                              fullband=fb, df=False, hop=HOP)
                m.eval()
                mpf = count_macs(m)
            except AssertionError as e:
                if fb:
                    notes.append(f"{nb} bands + fullband: {e}".split("\n")[0])
                    continue
                raise
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            mmacs = mpf * FRAME_RATE / 1e6
            pct = 100 * mmacs / CORE_MMACS
            if mmacs > CORE_MMACS:
                v = "DOES NOT FIT one core"
            elif mmacs > PS_MMACS:
                v = "fits the chip, BREAKS the PS cap"
            else:
                v = "fits both"
            tag = f"{nb} bands" + (" + fullband" if fb else "")
            print(f"{tag:28s}{params:10,}{mpf:12,.0f}{mmacs:10.1f}{pct:8.1f}%   {v}")
            break

    if notes:
        print("\nThe full-band branch is hardwired to the 48-band encoder width:")
        for n in notes:
            print(f"  · {n}")
        print("  Those rows are shown without it. Changing the band count is "
              "therefore an\n  architecture edit, not a flag -- the full-band "
              "branch has to be re-sized too.")

    print("\nNote: the ERB matrix is fixed and the convolutions are "
          "channel-wise, so\nparameters do not move with the band count. "
          "Only compute does.")
    print("A per-bin phase head below a cutoff adds a small decoder branch on "
          "top of\nthese figures; count it once the branch exists rather than "
          "estimating it here.")


if __name__ == "__main__":
    main()
