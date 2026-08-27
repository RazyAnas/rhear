"""The model must be strictly causal.

A model trained with any future context cannot be made streaming later without
retraining, so this is checked now rather than after a long run. The test
perturbs a FUTURE frame and asserts that no earlier output changes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
import torch
from gtcrn_lite import GTCRNLite, N_BINS, count_params, count_macs_per_frame, FS, HOP, ALG_LATENCY_MS


def test_strictly_causal():
    torch.manual_seed(0)
    m = GTCRNLite().eval()
    T, t_break = 30, 15
    x = torch.randn(1, 2, T, N_BINS)
    y = torch.randn(1, 2, T, N_BINS)
    y[:, :, :t_break] = x[:, :, :t_break]          # identical past, different future
    with torch.no_grad():
        a = m(x)[0][:, :, :t_break]
        b = m(y)[0][:, :, :t_break]
    d = (a - b).abs().max().item()
    assert d < 1e-5, f"future leaked into the past: max delta {d:.2e}"
    return d


def test_budgets():
    m = GTCRNLite().eval()
    p = count_params(m)
    macs, _ = count_macs_per_frame(m)
    rate = macs * (FS / HOP) / 1e6
    assert p <= 100_000, f"{p} params > 100k"
    assert rate <= 60.0, f"{rate:.1f} MMAC/s > 60"
    assert ALG_LATENCY_MS <= 8.0, f"{ALG_LATENCY_MS} ms > 8 ms"
    return p, rate, ALG_LATENCY_MS


def test_mask_is_bounded():
    """|mask| must be bounded so the network cannot invent energy -- important
    when a 160 dB transient arrives."""
    torch.manual_seed(1)
    m = GTCRNLite().eval()
    with torch.no_grad():
        mm, _, spp = m(torch.randn(1, 2, 20, N_BINS) * 500.0)
    # tanh saturates to exactly 1.0 in float32 for large inputs. The property
    # that matters is |mask| <= 1, i.e. the output magnitude can never exceed the
    # input magnitude, so the network cannot invent energy under a transient.
    assert mm.max().item() <= 1.0 + 1e-6, f"mask magnitude {mm.max().item():.4f} > 1"
    assert 0.0 <= spp.min().item() and spp.max().item() <= 1.0
    return mm.max().item()


if __name__ == "__main__":
    d = test_strictly_causal()
    print(f"  strictly causal ............... PASS (future->past delta {d:.2e})")
    p, r, l = test_budgets()
    print(f"  budgets ....................... PASS ({p:,} params, {r:.1f} MMAC/s, {l:.1f} ms)")
    b = test_mask_is_bounded()
    print(f"  mask bounded under 500x input . PASS (max |mask| {b:.4f} <= 1)")
