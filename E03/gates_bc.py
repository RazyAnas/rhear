#!/usr/bin/env python3
"""Pre-training gates for the 96-band + per-bin-phase variant.

An oracle ceiling says a perfect model of this family would pass. It says
nothing about whether THIS network, with THIS loss and THIS optimiser, can get
there. The extraction-ratio argument used so far ("G7-base reached 53% of its
PESQ ceiling, so...") is a heuristic from a different architecture. These are
direct measurements on the actual model.

  GATE A  export           ONNX export path still works
  GATE B  optimisation     forward equivalence to G7-base before any training
                           finite loss, non-zero gradient in every branch,
                           especially the phase head
                           tiny-set overfit
  GATE C  learnability     can the phase head learn the ORACLE phase target?

Gate C is the one that matters most. It removes every confound: the target is
known and exactly representable, so if the head cannot move toward it, the
problem is the architecture or the gradient path, not the data or the schedule.

    /opt/anaconda3/bin/python E03/gates_bc.py
"""

import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import N_FFT, FS                                   # noqa: E402
from gtcrn_phase import GTCRNPhase, enhance_phase, PHASE_BINS      # noqa: E402
from train_interim import Pairs, enhance, stft, loss_fn, si_sdr, DEV  # noqa: E402
from rhear_data.manifest import read_manifest                      # noqa: E402
from eval_stratified import load_model                             # noqa: E402
from oracle_ladder import score                                    # noqa: E402

CUT_HZ = 4000
results = {}
gates = {}


def gate(name, passed, detail=""):
    gates[name] = bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def load_g7_into(variant, ckpt):
    """G7-base's weights are shape-compatible: the band count changes only the
    fixed ERB buffer and the strides, never a weight tensor. So the variant can
    be initialised from G7-base directly, which is the whole point of the
    identity-initialised phase head."""
    sd = torch.load(ckpt, map_location=DEV)
    own = variant.state_dict()
    loaded, skipped = [], []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    variant.load_state_dict(own)
    return loaded, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
    ap.add_argument("--eval-data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--train-data", default=os.path.join(HERE, "..", "handoff", "data", "h3_20k"))
    ap.add_argument("--overfit-clips", type=int, default=16)
    ap.add_argument("--overfit-steps", type=int, default=400)
    ap.add_argument("--oracle-steps", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g012", "gates_bc.json"))
    a = ap.parse_args()

    torch.manual_seed(0)
    win = torch.hann_window(N_FFT, device=DEV)
    cut_bin = int(round(CUT_HZ / (FS / N_FFT)))

    print(f"\ndevice    {DEV}")
    print(f"variant   96 ERB bands + per-bin phase over {PHASE_BINS} bins "
          f"({PHASE_BINS * FS / N_FFT / 1000:.1f} kHz)")
    print(f"init from {a.ckpt}\n")

    model = GTCRNPhase().to(DEV)
    loaded, skipped = load_g7_into(model, a.ckpt)

    # ---------------------------------------------------------------- GATE A
    print("GATE A — export")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gate("G7-base weights load into the variant",
         len(skipped) <= 2, f"{len(loaded)} tensors loaded, skipped {skipped}")
    try:
        import onnx  # noqa: F401
        path = "/tmp/gtcrn_phase.onnx"
        torch.onnx.export(model.cpu().eval(), torch.randn(1, 2, 16, 257), path,
                          input_names=["spec"],
                          output_names=["mask", "cos", "sin", "spp"],
                          opset_version=17, dynamo=False)
        sz = os.path.getsize(path)
        gate("ONNX export", sz > 0, f"{sz / 1024:.0f} KB at opset 17")
    except Exception as e:                                   # noqa: BLE001
        gate("ONNX export", False, f"{type(e).__name__}: {str(e)[:120]}")
    model = model.to(DEV)

    # ---------------------------------------------------------------- GATE B1
    print("\nGATE B1 — forward equivalence: did the surgery destroy G7-base?")
    hdr, rows = read_manifest(os.path.join(a.eval_data, "manifest.jsonl"))
    ds = Pairs(a.eval_data, "test", rows, cache=False)
    test_rows = [r for r in rows if r.get("split") == "test"]
    near0 = [i for i, r in enumerate(test_rows) if abs(float(r["snr_db"])) <= 2.5]

    base, _ = load_model(a.ckpt)
    base.eval()
    model.eval()
    acc = {"noisy": [], "g7": [], "new": []}
    with torch.no_grad():
        for i in near0:
            x, y = ds[i]
            xb = x[None].to(DEV)
            clean, noisy = y.numpy(), x.numpy()
            _, g7, _, _ = enhance(base, xb, win)
            _, nw, _, _ = enhance_phase(model, xb, win)
            acc["noisy"].append(score(clean, noisy, noisy))
            acc["g7"].append(score(clean, noisy, g7[0].cpu().numpy()))
            acc["new"].append(score(clean, noisy, nw[0].cpu().numpy()))

    def agg(k):
        d = {m: float(np.nanmean([r[m] for r in acc[k]])) for m in ("stoi", "pesq", "sdr")}
        d["dsisdr"] = d["sdr"] - float(np.nanmean([r["sdr"] for r in acc["noisy"]]))
        return d

    A = {k: agg(k) for k in acc}
    results["forward_equivalence"] = A
    print(f"  {'':10s}{'STOI':>8s}{'PESQ':>8s}{'dSI-SDR':>10s}")
    for k, label in (("noisy", "input"), ("g7", "G7-base"), ("new", "96+phase")):
        print(f"  {label:10s}{A[k]['stoi']:8.3f}{A[k]['pesq']:8.3f}{A[k]['dsisdr']:10.2f}")
    drop = A["g7"]["dsisdr"] - A["new"]["dsisdr"]
    print(f"\n  The untrained variant is NOT expected to pass. What this "
          f"decides is whether\n  the variant can be FINE-TUNED from G7-base or "
          f"must be trained from scratch:\n  dSI-SDR moved {-drop:+.2f} dB.")
    gate("variant can be warm-started from G7-base",
         A["new"]["dsisdr"] > 0.5 * A["g7"]["dsisdr"],
         f"{A['new']['dsisdr']:.2f} vs {A['g7']['dsisdr']:.2f} dB — "
         f"a fail here means budget a from-scratch run, not a fine-tune")

    # ---------------------------------------------------------------- GATE B2
    print("\nGATE B2 — is there a usable gradient, and does it reach the phase head?")
    hdr2, rows2 = read_manifest(os.path.join(a.train_data, "manifest.jsonl"))
    tds = Pairs(a.train_data, "train", rows2, cache=False)
    idx = list(range(a.overfit_clips))
    xb = torch.stack([tds[i][0] for i in idx]).to(DEV)
    yb = torch.stack([tds[i][1] for i in idx]).to(DEV)

    model.train()
    model.zero_grad()
    S_est, wav, X, spp = enhance_phase(model, xb, win)
    S_ref = stft(yb, win, int(model.hop_.item()))
    total, parts = loss_fn(wav, yb, S_est, S_ref, spp)
    gate("loss is finite", bool(torch.isfinite(total)), f"{float(total):.4f}")
    total.backward()

    branches = {"erb/enc": "enc.", "fullband": "fb.", "fuse": "fuse",
                "dprnn": "dprnn.", "decoder": "dec.", "spp": "spp.",
                "phase head": "phase_head."}
    norms, bad = {}, 0
    for label, pre in branches.items():
        g = [p.grad for n, p in model.named_parameters()
             if n.startswith(pre) and p.grad is not None]
        if not g:
            norms[label] = 0.0
            continue
        norms[label] = float(torch.sqrt(sum((q ** 2).sum() for q in g)))
        bad += sum(int((~torch.isfinite(q)).sum()) for q in g)
    results["grad_norms"] = norms
    for label, v in norms.items():
        print(f"    {label:12s} grad norm {v:12.4e}")
    gate("no NaN or Inf gradients", bad == 0, f"{bad} non-finite entries")
    gate("phase head has a non-zero gradient", norms["phase head"] > 0,
         f"{norms['phase head']:.3e}")
    ratio = norms["phase head"] / max(norms["decoder"], 1e-12)
    gate("phase head gradient is not vanishing vs the decoder",
         ratio > 1e-3, f"phase/decoder = {ratio:.3f}")

    # ---------------------------------------------------------------- GATE B3
    print(f"\nGATE B3 — can it overfit {a.overfit_clips} clips in "
          f"{a.overfit_steps} steps?")
    m2 = GTCRNPhase().to(DEV)
    load_g7_into(m2, a.ckpt)
    m2.train()
    opt = torch.optim.Adam(m2.parameters(), lr=5e-4)
    hist = []
    t0 = time.time()
    for s in range(a.overfit_steps):
        opt.zero_grad()
        S_est, wav, X, spp = enhance_phase(m2, xb, win)
        S_ref = stft(yb, win, int(m2.hop_.item()))
        total, _ = loss_fn(wav, yb, S_est, S_ref, spp)
        if not torch.isfinite(total):
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(), 5.0)
        opt.step()
        if s % max(1, a.overfit_steps // 8) == 0 or s == a.overfit_steps - 1:
            with torch.no_grad():
                sd = float(si_sdr(wav, yb).mean())
            hist.append({"step": s, "loss": float(total), "sisdr": sd})
            print(f"    step {s:5d}   loss {float(total):9.4f}   "
                  f"SI-SDR {sd:7.2f} dB")
    results["overfit"] = hist
    print(f"    ({time.time() - t0:.0f}s)")
    if len(hist) >= 2:
        gate("training loss falls substantially",
             hist[-1]["loss"] < 0.6 * hist[0]["loss"],
             f"{hist[0]['loss']:.3f} -> {hist[-1]['loss']:.3f}")
        gate("SI-SDR on the memorised set climbs",
             hist[-1]["sisdr"] > hist[0]["sisdr"] + 3.0,
             f"{hist[0]['sisdr']:.2f} -> {hist[-1]['sisdr']:.2f} dB")
    else:
        gate("training loss falls substantially", False, "loss went non-finite")

    # ---------------------------------------------------------------- GATE C
    print(f"\nGATE C — can the phase head learn the ORACLE phase target?")
    print("  Everything except the phase head is frozen. The target is the "
          "CLEAN phase,\n  which is exactly what the oracle row substitutes. If "
          "this does not converge,\n  the 24 dB ceiling is unreachable no matter "
          "what the data looks like.\n")
    m3 = GTCRNPhase().to(DEV)
    load_g7_into(m3, a.ckpt)
    for n, p in m3.named_parameters():
        p.requires_grad = n.startswith("phase_head.")
    m3.train()
    opt3 = torch.optim.Adam([p for p in m3.parameters() if p.requires_grad], lr=1e-3)

    hop = int(m3.hop_.item())
    with torch.no_grad():
        Xb = stft(xb, win, hop)
        Sb = stft(yb, win, hop)
        # The rotation that takes noisy phase to clean phase, per bin, as a
        # unit vector -- the exact quantity the head emits.
        d = (Sb / (Sb.abs() + 1e-9)) * torch.conj(Xb / (Xb.abs() + 1e-9))
        tgt_cos = d.real.transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        tgt_sin = d.imag.transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        # Weight by clean magnitude: phase in a silent bin is undefined and
        # fitting it is noise. This is what a real phase loss would have to do.
        w = Sb.abs().transpose(1, 2).unsqueeze(1)[..., :PHASE_BINS]
        w = w / (w.mean() + 1e-9)

    chist = []
    for s in range(a.oracle_steps):
        opt3.zero_grad()
        spec = torch.stack([Xb.real, Xb.imag], 1).transpose(2, 3)
        _, (cos, sin), _ = m3(spec)
        l = (w * ((cos - tgt_cos) ** 2 + (sin - tgt_sin) ** 2)).mean()
        l.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in m3.parameters() if p.requires_grad], 5.0)
        opt3.step()
        if s % max(1, a.oracle_steps // 8) == 0 or s == a.oracle_steps - 1:
            with torch.no_grad():
                # cosine of the residual angle: 1.0 = phase exactly recovered,
                # 0.0 = no better than leaving the noisy phase alone.
                align = float((w * (cos * tgt_cos + sin * tgt_sin)).sum() / w.sum())
                base_align = float((w * tgt_cos).sum() / w.sum())
            chist.append({"step": s, "loss": float(l), "align": align})
            print(f"    step {s:5d}   loss {float(l):9.5f}   "
                  f"phase alignment {align:+.4f}   (identity = {base_align:+.4f})")
    results["oracle_phase"] = {"history": chist, "identity_alignment": base_align}
    if chist:
        gained = chist[-1]["align"] - base_align
        gate("phase head moves toward the oracle target",
             chist[-1]["align"] > base_align + 0.02,
             f"alignment {base_align:+.4f} -> {chist[-1]['align']:+.4f} "
             f"({gained:+.4f})")
        gate("oracle-target loss falls", chist[-1]["loss"] < 0.8 * chist[0]["loss"],
             f"{chist[0]['loss']:.5f} -> {chist[-1]['loss']:.5f}")

    # ---------------------------------------------------------------- summary
    n, t = sum(gates.values()), len(gates)
    print(f"\n{'=' * 66}")
    print(f"{n}/{t} gates passed")
    failed = [k for k, v in gates.items() if not v]
    if failed:
        print("FAILED:")
        for k in failed:
            print(f"  · {k}")
        print("\nDo not spend the 6-epoch run until these are understood.")
    else:
        print("Gates A, B and C are green. The architecture can represent the\n"
              "correction, the gradient reaches it, and the optimiser moves it\n"
              "toward the oracle target. Gate D (generalisation) is now worth\n"
              "the compute.")
    print("=" * 66)

    with open(a.out, "w") as f:
        json.dump({"ckpt": a.ckpt, "gates": gates, "results": results}, f, indent=1)
    print(f"\nwritten   {a.out}")


if __name__ == "__main__":
    main()
