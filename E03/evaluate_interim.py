#!/usr/bin/env python3
"""Score the trained L1 on the held-out test split and export A/B examples.

INTERIM: the noise is synthesised, so these numbers are NOT a G3 result. Every
output file and the JSON summary carry that label.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import GTCRNLite, N_FFT, FS, count_params, count_macs_per_frame, HOP
from train_interim import enhance, si_sdr, Pairs, DEV
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn


def score(clean, noisy, enh, fs=FS):
    out = {}
    for tag, sig in (("noisy", noisy), ("enhanced", enh)):
        try:
            out[f"stoi_{tag}"] = float(stoi_fn(clean, sig, fs, extended=False))
        except Exception:
            out[f"stoi_{tag}"] = float("nan")
        try:
            out[f"pesq_{tag}"] = float(pesq_fn(fs, clean, sig, "wb"))
        except Exception:
            out[f"pesq_{tag}"] = float("nan")
    c = torch.from_numpy(clean)[None]
    out["sisdr_noisy"] = float(si_sdr(torch.from_numpy(noisy)[None], c))
    out["sisdr_enhanced"] = float(si_sdr(torch.from_numpy(enh)[None], c))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="runs/interim/best.pt")
    ap.add_argument("--out", default="runs/interim/eval")
    ap.add_argument("--n-examples", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    model = GTCRNLite().to(DEV)
    model.load_state_dict(torch.load(a.ckpt, map_location=DEV))
    model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    print(f"  {len(ds)} test clips  |  ckpt {a.ckpt}  |  device {DEV}")

    res, per_class = [], {}
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        e = est[0].cpu().numpy().astype(np.float64)
        c = y.numpy().astype(np.float64)
        n = x.numpy().astype(np.float64)
        s = score(c, n, e)
        r = ds.rows[i]
        s.update(id=r["id"], snr_db=r["snr_db"],
                 stationarity=r["mixture_stationarity"],
                 classes=[l["cls"] for l in r["noise_layers"]])
        res.append(s)
        per_class.setdefault(r["mixture_stationarity"], []).append(s)
        if i % 50 == 0:
            print(f"    {i}/{len(ds)}")

    def agg(rs, k):
        v = [r[k] for r in rs if np.isfinite(r[k])]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "WARNING": "INTERIM — real speech, SYNTHESISED noise. NOT a G3 result.",
        "checkpoint": a.ckpt, "n_test": len(res),
        "model": {"params": count_params(model),
                  "mmacs": count_macs_per_frame(model)[0] * (FS / HOP) / 1e6,
                  "latency_ms": 8.0},
        "overall": {k: round(agg(res, k), 4) for k in
                    ("stoi_noisy", "stoi_enhanced", "pesq_noisy", "pesq_enhanced",
                     "sisdr_noisy", "sisdr_enhanced")},
        "by_stationarity": {c: {k: round(agg(v, k), 4) for k in
                                ("stoi_noisy", "stoi_enhanced", "pesq_noisy",
                                 "pesq_enhanced", "sisdr_noisy", "sisdr_enhanced")}
                            for c, v in per_class.items()},
    }
    o = summary["overall"]
    summary["deltas"] = {
        "d_stoi": round(o["stoi_enhanced"] - o["stoi_noisy"], 4),
        "d_pesq": round(o["pesq_enhanced"] - o["pesq_noisy"], 4),
        "d_sisdr": round(o["sisdr_enhanced"] - o["sisdr_noisy"], 3),
    }

    # A/B examples spanning the SNR range, exported for the dashboard
    idx = np.argsort([r["snr_db"] for r in res])
    picks = [int(idx[k]) for k in np.linspace(0, len(idx) - 1, a.n_examples).astype(int)]
    ex = []
    for j, i in enumerate(picks):
        x, y = ds[i]
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        e = est[0].cpu().numpy()
        base = f"ex{j:02d}"
        for tag, sig in (("noisy", x.numpy()), ("clean", y.numpy()), ("enhanced", e)):
            sf.write(os.path.join(a.out, f"{base}_{tag}.wav"),
                     np.clip(sig, -1, 1).astype(np.float32), FS)
        ex.append(dict(base=base, **{k: res[i][k] for k in
                                     ("id", "snr_db", "stationarity", "classes",
                                      "stoi_noisy", "stoi_enhanced",
                                      "pesq_noisy", "pesq_enhanced",
                                      "sisdr_noisy", "sisdr_enhanced")}))
    summary["examples"] = ex
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    print("\n  INTERIM RESULTS (real speech, SYNTHESISED noise - not G3)")
    print("  " + "=" * 58)
    print(f"  {'metric':<12}{'noisy':>10}{'enhanced':>11}{'delta':>10}   PS target")
    for m, tgt in (("stoi", "> 0.85"), ("pesq", "> 2.5"), ("sisdr", "> 15 dB")):
        a_, b_ = o[f"{m}_noisy"], o[f"{m}_enhanced"]
        print(f"  {m.upper():<12}{a_:10.3f}{b_:11.3f}{b_-a_:+10.3f}   {tgt}")
    print("  " + "=" * 58)
    for c, v in summary["by_stationarity"].items():
        print(f"  {c:<16} STOI {v['stoi_noisy']:.3f}->{v['stoi_enhanced']:.3f}   "
              f"PESQ {v['pesq_noisy']:.2f}->{v['pesq_enhanced']:.2f}   "
              f"SI-SDR {v['sisdr_noisy']:+.1f}->{v['sisdr_enhanced']:+.1f} dB")
    print(f"\n  {a.n_examples} A/B examples -> {a.out}")
