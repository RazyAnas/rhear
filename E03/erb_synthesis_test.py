#!/usr/bin/env python3
"""Training-free test: rectangular vs triangular ERB synthesis.

ERBSplit.inverse spreads each band's gain to its bins with a BINARY matrix
(Minv = (M > 0)), i.e. piecewise-constant. Every band edge is a step
discontinuity in the applied gain, and at 48 bands over 257 bins the widest
bands are ~19 bins across.

PercepNet (Valin 2020) and DeepFilterNet (Schroeter 2022) both use overlapping
triangular filterbanks for exactly this reason. This swaps in a triangular
synthesis -- linear interpolation between adjacent band centres in ERB space --
and changes NOTHING else. The model, its weights and its outputs are identical;
only the band->bin spreading differs.

Costs nothing extra on the MCU: still a sparse matmul, 2 non-zeros per row
instead of 1.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import FS, N_BINS
from train_interim import enhance, Pairs, DEV
from rhear_data.manifest import read_manifest
from eval_stratified import load_model, score_one, agg


def triangular_Minv(n_bins=N_BINS, n_bands=48, fs=FS):
    """(n_bins, n_bands) linear-interpolation synthesis, rows sum to 1."""
    hz = torch.linspace(0, fs / 2, n_bins)
    erb = 21.4 * torch.log10(1 + 0.00437 * hz)
    edges = torch.linspace(erb[0], erb[-1], n_bands + 1)
    centres = (edges[:-1] + edges[1:]) / 2                  # band centre in ERB
    W = torch.zeros(n_bins, n_bands)
    for f in range(n_bins):
        e = erb[f]
        if e <= centres[0]:
            W[f, 0] = 1.0
        elif e >= centres[-1]:
            W[f, -1] = 1.0
        else:
            g = int(torch.searchsorted(centres, e).item()) - 1
            g = max(0, min(g, n_bands - 2))
            span = (centres[g + 1] - centres[g]).clamp_min(1e-9)
            a = ((e - centres[g]) / span).clamp(0, 1)
            W[f, g] = 1.0 - a
            W[f, g + 1] = a
    return W


def run(model, ds, meta, win):
    out = []
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        c, n = y.numpy(), x.numpy()
        e = est[0].cpu().numpy()
        L = min(len(c), len(n), len(e))
        r = dict(meta[i]); r.update(score_one(c[:L], n[:L], e[:L]))
        out.append(r)
    return agg(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/g2_fullband/best.pt")
    ap.add_argument("--sets", nargs="+", default=["edef", "realnoise"])
    ap.add_argument("--out", default="runs/g012/erb_synthesis.json")
    a = ap.parse_args()

    model, _ = load_model(a.ckpt)
    win = torch.hann_window(512, device=DEV)
    rect = model.erb.Minv.clone()
    tri = triangular_Minv(model.erb.n_bins, model.erb.n_bands).to(DEV)

    nz_r = float((rect > 0).float().sum(1).mean())
    nz_t = float((tri > 1e-6).float().sum(1).mean())
    print("  synthesis matrices: rectangular %.2f non-zeros/bin, "
          "triangular %.2f non-zeros/bin" % (nz_r, nz_t))
    print("  ckpt %s\n" % a.ckpt)

    res = {}
    for s in a.sets:
        d = os.path.join(HERE, "..", "handoff", "data", s)
        hdr, rows = read_manifest(os.path.join(d, "manifest.jsonl"))
        ds = Pairs(d, "test", rows, cache=False)
        meta = [r for r in rows if r["split"] == "test"]
        print("  === %s (%d clips) ===" % (s, len(ds)))
        res[s] = {}
        for tag, M in (("rectangular (current)", rect), ("triangular (new)", tri)):
            model.erb.Minv = M
            g = run(model, ds, meta, win)
            res[s][tag] = g
            print("    %-22s STOI %.4f  PESQ %.4f  SI-SDR %6.2f  "
                  "SI-SIR %6.2f  SI-SAR %6.2f"
                  % (tag, g["stoi_e"], g["pesq_e"], g["sisdr_e"], g["sir_e"], g["sar_e"]))
        r, t = res[s]["rectangular (current)"], res[s]["triangular (new)"]
        print("    %-22s      %+.4f       %+.4f         %+.2f        %+.2f        %+.2f"
              % ("DELTA", t["stoi_e"] - r["stoi_e"], t["pesq_e"] - r["pesq_e"],
                 t["sisdr_e"] - r["sisdr_e"], t["sir_e"] - r["sir_e"],
                 t["sar_e"] - r["sar_e"]))
        print()
    os.makedirs(os.path.dirname(os.path.join(HERE, a.out)), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, a.out), "w"), indent=1)
    print("  -> %s" % a.out)
