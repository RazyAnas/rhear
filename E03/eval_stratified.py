#!/usr/bin/env python3
"""Score a checkpoint on an eval set, stratified.

    python3 eval_stratified.py --data <eval_dir> --ckpt <best.pt> \
                                 --out <dir> [--tag name]

Reports STOI / PESQ / SI-SDR overall AND by input-SNR bucket, noise
class, and reverberation. An overall average alone hides where a model
passes and where it fails: the same 300 clips that averaged STOI 0.821
were already above 0.90 wherever input SNR was >= 5 dB, and the whole
deficit sat in the negative-SNR clips.

Noise class is per LAYER -- a mixture carries 1-3 noise layers, so a
clip counts toward each class present in it.
"""
import os, sys, json, argparse, collections, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import GTCRNLite, FS
from train_interim import enhance, si_sdr, Pairs, DEV
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn
from phase5_comb import si_metrics

SNR_BUCKETS = [(-99, -5, "< -5 dB"), (-5, 0, "-5..0 dB"), (0, 5, "0..5 dB"),
                (5, 10, "5..10 dB"), (10, 15, "10..15 dB"), (15, 99, "> 15 dB")]


def load_model(ckpt):
    """Load a checkpoint into the right variant.

    The phase-free model has 1 output channel in the final decoder block
    instead of 2, so constructing GTCRNLite() with its default phase=True
    cannot load it. Detect from the shape rather than from a filename.
    """
    sd = torch.load(ckpt, map_location=DEV)
    k = "dec.3.pw.weight"
    has_phase = sd[k].shape[0] == 2 if k in sd else True
    has_fb = any(q.startswith("fb.") for q in sd)      # G2 full-band branch
    has_df = any(q.startswith("dfdec.") for q in sd)   # G6 deep-filter stage
    taps = sd["dfdec.2.pw.weight"].shape[0] // 2 if has_df else 5
    # Checkpoints from before hop became configurable have no hop_ buffer.
    # They were all trained at 64, so inject it rather than loading loosely --
    # strict loading is what catches every OTHER kind of mismatch.
    if "hop_" not in sd:
        sd["hop_"] = torch.tensor(64)
    hop = int(sd["hop_"].item())
    ch = (sd["enc.0.pw.weight"].shape[0], sd["enc.1.pw.weight"].shape[0],
          sd["enc.2.pw.weight"].shape[0], sd["enc.3.pw.weight"].shape[0])
    # The band count is a buffer, so it travels with the weights. Reading it
    # here is what lets a 96-band checkpoint load without every caller knowing
    # about it; a default of 48 would fail on erb.M's shape instead.
    n_bands = sd["erb.M"].shape[0] if "erb.M" in sd else 48
    m = GTCRNLite(ch=ch, n_bands=n_bands, phase=has_phase, fullband=has_fb,
                  df=has_df, df_taps=taps, hop=hop).to(DEV)
    m.load_state_dict(sd)
    m.eval()
    return m, (has_phase, has_fb)


def score_one(clean, noisy, enh):
    o = {}
    # SI-SIR (noise removed) and SI-SAR (damage caused). SI-SDR alone cannot
    # separate them, and for L1 the binding constraint is SAR, not SIR.
    _, o["sir_e"], o["sar_e"] = si_metrics(enh, clean, noisy - clean)
    for tag, sig in (("n", noisy), ("e", enh)):
        try:
            o["stoi_" + tag] = float(stoi_fn(clean, sig, FS, extended=False))
        except Exception:
            o["stoi_" + tag] = float("nan")
        try:
            o["pesq_" + tag] = float(pesq_fn(FS, clean, sig, "wb"))
        except Exception:
            o["pesq_" + tag] = float("nan")
    c = torch.from_numpy(clean)[None]
    o["sisdr_n"] = float(si_sdr(torch.from_numpy(noisy)[None], c))
    o["sisdr_e"] = float(si_sdr(torch.from_numpy(enh)[None], c))
    return o


def agg(rows):
    if not rows:
        return None
    f = lambda k: float(np.nanmean([r[k] for r in rows]))
    return dict(n=len(rows),
                stoi_n=f("stoi_n"), stoi_e=f("stoi_e"),
                pesq_n=f("pesq_n"), pesq_e=f("pesq_e"),
                sisdr_n=f("sisdr_n"), sisdr_e=f("sisdr_e"),
                sir_e=f("sir_e"), sar_e=f("sar_e"))


def line(label, a):
    if not a:
        return "  %-16s      (no clips)" % label
    return ("  %-16s n=%4d  STOI %.3f->%.3f (%+.3f)  "
            "PESQ %.2f->%.2f (%+.2f)  SI-SDR %+5.1f (%+.1f)  "
            "SIR %+5.1f  SAR %+5.1f"
            % (label, a["n"], a["stoi_n"], a["stoi_e"],
               a["stoi_e"] - a["stoi_n"], a["pesq_n"], a["pesq_e"],
               a["pesq_e"] - a["pesq_n"], a["sisdr_e"],
               a["sisdr_e"] - a["sisdr_n"], a["sir_e"], a["sar_e"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or os.path.basename(a.data.rstrip("/"))

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, a.split, rows, cache=False)
    meta = [r for r in rows if r["split"] == a.split]
    assert len(ds) == len(meta), "manifest/dataset mismatch"

    model, (has_phase, has_fb) = load_model(a.ckpt)
    print("  model: %d params | hop %d | phase branch %s | full-band branch %s"
          % (sum(p.numel() for p in model.parameters()),
             int(model.hop_.item()),
             "PRESENT" if has_phase else "REMOVED",
             "PRESENT" if has_fb else "absent"))
    win = torch.hann_window(512, device=DEV)
    print("  %s | %d clips | ckpt %s" % (tag, len(ds), a.ckpt))

    out = []
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        c, n = y.numpy(), x.numpy()
        e = est[0].cpu().numpy()
        L = min(len(c), len(n), len(e))
        r = dict(meta[i])
        r.update(score_one(c[:L], n[:L], e[:L]))
        out.append(r)
        if (i + 1) % 60 == 0:
            print("    %d/%d" % (i + 1, len(ds)), flush=True)

    res = {"tag": tag, "ckpt": a.ckpt, "n": len(out),
           "overall": agg(out), "by_snr": {}, "by_class": {}, "by_reverb": {}}

    print("\n=== %s ===" % tag)
    print(line("OVERALL", res["overall"]))

    print("\n--- by input SNR ---")
    for lo, hi, lab in SNR_BUCKETS:
        sel = [r for r in out if lo <= r.get("snr_db", -999) < hi]
        res["by_snr"][lab] = agg(sel)
        print(line(lab, res["by_snr"][lab]))

    print("\n--- by noise class (per layer; a clip counts in each) ---")
    classes = sorted({l["cls"] for r in out
                       for l in r.get("noise_layers", [])})
    for k in classes:
        sel = [r for r in out
               if any(l["cls"] == k for l in r.get("noise_layers", []))]
        res["by_class"][k] = agg(sel)
        print(line(k, res["by_class"][k]))

    print("\n--- by reverberation ---")
    for k in sorted({str(r.get("rir_kind")) for r in out}):
        sel = [r for r in out if str(r.get("rir_kind")) == k]
        res["by_reverb"][k] = agg(sel)
        print(line(k, res["by_reverb"][k]))

    p = os.path.join(a.out, "eval_%s.json" % tag)
    json.dump(res, open(p, "w"), indent=1)
    print("\n  -> %s" % p)
