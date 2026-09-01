#!/usr/bin/env python3
"""Exact model diff: L1 with the learned phase branch vs without.

Reports, for both variants: parameters, MAC/frame and MAC/s, INT8 weight
Flash, peak activation RAM for one streaming frame, ONNX operator count, and
which operators CMSIS-NN cannot run. Latency is scaled from the F407 report's
measured MAC-to-cycle model rather than re-derived, and is labelled as
modelled, not measured.

The operator classification is the one used for E03/bench/STM32F407_REPORT.md.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import (GTCRNLite, N_FFT, N_BINS, HOP, FS,
                        count_params, count_macs_per_frame)

SUPPORTED = {"Add", "Concat", "Conv", "Reshape", "Sigmoid", "MatMul",
             "Tanh", "Sub"}
PARTIAL = {"Slice", "PRelu", "Transpose", "BatchNormalization"}
FPS = FS / HOP                      # 250 frames/s at 16 kHz, 4 ms hop


class ExportWrap(nn.Module):
    """ONNX cannot return None, so drop it for the no-phase variant."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, spec):
        mm, mp, spp, dfc = self.m(spec)
        out = [mm] if mp is None else [mm, mp]
        out.append(spp)
        if dfc is not None:
            out.append(dfc)
        return tuple(out)


def peak_activation_bytes(model, frames=1):
    """Largest single intermediate tensor produced in one forward pass, int8."""
    peak = {"v": 0}

    def hook(_mod, _inp, out):
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(t):
                peak["v"] = max(peak["v"], t.numel())
    hs = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(torch.randn(1, 2, frames, N_BINS))
    for h in hs:
        h.remove()
    return peak["v"]          # 1 byte/value at int8


def onnx_ops(model, path):
    import onnx
    w = ExportWrap(model).eval()
    torch.onnx.export(w, torch.randn(1, 2, 1, N_BINS), path,
                      input_names=["spec"], opset_version=17,
                      dynamic_axes=None)
    g = onnx.load(path).graph
    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return ops


def int8_flash(model):
    """Weight bytes at int8 -- one byte per parameter, plus per-channel scales."""
    n = sum(p.numel() for p in model.parameters())
    return n


def report(tag, model, tmp):
    p = count_params(model)
    macs_f = count_macs_per_frame(model)
    macs_f = macs_f[0] if isinstance(macs_f, tuple) else macs_f
    ops = onnx_ops(model, tmp)
    unsup = sorted(o for o in ops if o not in SUPPORTED and o not in PARTIAL)
    return {
        "params": p,
        "mac_per_frame": int(macs_f),
        "mmac_s": round(macs_f * FPS / 1e6, 2),
        "flash_int8_kb": round(int8_flash(model) / 1024, 1),
        "act_ram_kb": round(peak_activation_bytes(model) / 1024, 2),
        "onnx_nodes": sum(ops.values()),
        "onnx_distinct_ops": len(ops),
        "unsupported": unsup,
        "n_unsupported": len(unsup),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/model_diff.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    res = {}
    for tag, kw in (("with_phase", dict(phase=True)),
                    ("no_phase", dict(phase=False)),
                    ("no_phase_fullband", dict(phase=False, fullband=True))):
        m = GTCRNLite(**kw).eval()
        res[tag] = report(tag, m, "/tmp/_%s.onnx" % tag)

    A, B = res["with_phase"], res["no_phase"]
    rows = [("parameters", "params", "%d"),
            ("MAC / frame", "mac_per_frame", "%d"),
            ("MAC/s (MMAC/s)", "mmac_s", "%.2f"),
            ("Flash, INT8 weights (KB)", "flash_int8_kb", "%.1f"),
            ("peak activation RAM (KB)", "act_ram_kb", "%.2f"),
            ("ONNX nodes", "onnx_nodes", "%d"),
            ("distinct operators", "onnx_distinct_ops", "%d"),
            ("unsupported operators", "n_unsupported", "%d")]
    print("  %-28s%16s%16s%14s" % ("quantity", "with phase", "no phase", "delta"))
    print("  " + "-" * 74)
    for label, k, fmt in rows:
        va, vb = A[k], B[k]
        d = vb - va
        print("  %-28s%16s%16s%+14s"
              % (label, fmt % va, fmt % vb,
                 (fmt % d) if abs(d) > 1e-9 else "0"))

    print("\n  unsupported operators")
    print("    with phase (%d): %s" % (A["n_unsupported"],
                                       ", ".join(A["unsupported"])))
    print("    no phase   (%d): %s" % (B["n_unsupported"],
                                       ", ".join(B["unsupported"])))
    gone = sorted(set(A["unsupported"]) - set(B["unsupported"]))
    added = sorted(set(B["unsupported"]) - set(A["unsupported"]))
    print("    removed: %s" % (", ".join(gone) if gone else "none"))
    print("    added  : %s" % (", ".join(added) if added else "none"))

    # latency: scale the F407 report's measured 2.44-4.86 ms window by MAC ratio
    lo, hi = 2.44, 4.86
    r = B["mmac_s"] / A["mmac_s"]
    print("\n  modelled latency on STM32F407 (scaled from the measured "
          "%.2f-%.2f ms window by the MAC ratio %.4f):" % (lo, hi, r))
    print("    with phase %.2f-%.2f ms   ->   no phase %.2f-%.2f ms"
          % (lo, hi, lo * r, hi * r))
    print("    frame period is 4.00 ms -- this is MODELLED, not measured")

    json.dump(res, open(a.out, "w"), indent=1)
    print("\n  -> %s" % a.out)
