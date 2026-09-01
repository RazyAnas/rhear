#!/usr/bin/env python3
"""G3 - structured pruning sensitivity. The PS names pruning; we had not tested it.

STRUCTURED ONLY. Unstructured sparsity buys nothing on an ESP32-S3, which has
no sparse kernels: zeroed weights still cost a multiply. Only removing whole
units reduces work.

Where the work actually is, from bench/model_diff.json:
    GRU            42.5 % of MACs
    ConvTranspose  32.7 %
    Conv            5.3 %
so GRU hidden units are the target worth pruning.

Method: mask the least-important hidden units on the FROZEN checkpoint and
measure the damage, with no fine-tuning. Masking is not the same as removing --
it does not save any compute here -- but it answers the only question that
matters first: is there redundancy to remove at all? If quality collapses at
10 % masked, the model has no slack and pruning is dead without ever building
a smaller one. If quality survives, then rebuilding physically smaller is
justified and the saving can be computed exactly.

Expected result, stated before running: low payoff. At 22,956 parameters we are
already half of GTCRN's 48.2 k, the efficiency reference this model was scaled
against.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import GTCRNLite, N_FFT, FS, count_params
from train_interim import Pairs, enhance, DEV
from rhear_data.manifest import read_manifest
from phase5_comb import si_metrics, score
from eval_stratified import load_model


def gru_unit_importance(gru, lin, bidir=False):
    """L2 importance per hidden unit: how much it receives and how much it sends.

    GRU packs 3 gates into each weight matrix, so hidden unit j owns rows
    j, H+j and 2H+j of weight_ih and weight_hh.
    """
    H = gru.hidden_size
    imp = []
    for d in ([""] if not bidir else ["", "_reverse"]):
        wih = getattr(gru, "weight_ih_l0" + d)
        whh = getattr(gru, "weight_hh_l0" + d)
        rows = torch.stack([wih[j::H].norm() + whh[j::H].norm()
                            for j in range(H)])
        imp.append(rows)
    return torch.stack(imp)          # (dirs, H)


def mask_gru_units(gru, lin, keep, bidir=False):
    """Zero the weights of hidden units not in `keep`, in place."""
    H = gru.hidden_size
    with torch.no_grad():
        for di, d in enumerate([""] if not bidir else ["", "_reverse"]):
            drop = [j for j in range(H) if not keep[di][j]]
            if not drop:
                continue
            wih = getattr(gru, "weight_ih_l0" + d)
            whh = getattr(gru, "weight_hh_l0" + d)
            bih = getattr(gru, "bias_ih_l0" + d, None)
            bhh = getattr(gru, "bias_hh_l0" + d, None)
            for j in drop:
                for g in range(3):
                    wih[g * H + j].zero_()
                    whh[g * H + j].zero_()
                    if bih is not None:
                        bih[g * H + j].zero_()
                    if bhh is not None:
                        bhh[g * H + j].zero_()
                whh[:, j].zero_()
                # the linear that reads the GRU output: drop that input column
                col = j + di * H
                if col < lin.weight.shape[1]:
                    lin.weight[:, col].zero_()


def prune(model, frac):
    """Mask the lowest-importance `frac` of hidden units in both GRUs."""
    if frac <= 0:
        return 0
    n_dropped = 0
    for gru, lin, bidir in ((model.dprnn.gru_f, model.dprnn.lin_f, True),
                            (model.dprnn.gru_t, model.dprnn.lin_t, False)):
        imp = gru_unit_importance(gru, lin, bidir)
        H = gru.hidden_size
        k = int(round(frac * H))
        keep = torch.ones_like(imp, dtype=torch.bool)
        if k > 0:
            for di in range(imp.shape[0]):
                idx = torch.argsort(imp[di])[:k]
                keep[di, idx] = False
                n_dropped += k
        mask_gru_units(gru, lin, keep, bidir)
    return n_dropped


def evaluate(model, ds, idx, meta):
    win = torch.hann_window(N_FFT, device=DEV)
    acc = {k: [] for k in ("stoi", "pesq", "sdr", "sir", "sar")}
    for i in idx:
        x, y = ds[int(i)]
        c = y.numpy().astype(np.float64)
        nz = x.numpy().astype(np.float64)
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        e = est[0].cpu().numpy().astype(np.float64)
        L = min(len(e), len(c))
        st, pq = score(c[:L], e[:L])
        sd, si, sa = si_metrics(e[:L], c[:L], (nz - c)[:L])
        for k, v in zip(("stoi", "pesq", "sdr", "sir", "sar"),
                        (st, pq, sd, si, sa)):
            acc[k].append(v)
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="runs/h9_frozen/h3_snapshot.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="runs/h9_frozen/g3_pruning.json")
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    meta = [r for r in rows if r["split"] == "test"]
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)

    m0, _ = load_model(a.ckpt)
    print("  %d clips | %s | %d params\n" % (len(idx), a.ckpt, count_params(m0)))
    print("  GRU hidden sizes: freq %d (bidirectional), time %d"
          % (m0.dprnn.gru_f.hidden_size, m0.dprnn.gru_t.hidden_size))
    print("  GRU is 42.5%% of MACs, so this is the structure worth pruning\n")

    print("  %-14s%8s%8s%9s%9s%9s%10s"
          % ("units masked", "STOI", "PESQ", "SI-SDR", "SI-SIR", "SI-SAR", "dPESQ"))
    print("  " + "-" * 72)
    res, base = {}, None
    for frac in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        m, _ = load_model(a.ckpt)             # reload clean every time
        nd = prune(m, frac)
        v = evaluate(m, ds, idx, meta)
        if base is None:
            base = v
        res["%.0f%%" % (100 * frac)] = dict(v, dropped=nd)
        print("  %-14s%8.3f%8.3f%9.2f%9.2f%9.2f%+10.3f"
              % ("%.0f%% (%d)" % (100 * frac, nd), v["stoi"], v["pesq"],
                 v["sdr"], v["sir"], v["sar"], v["pesq"] - base["pesq"]))

    json.dump({"baseline": base, "sweep": res}, open(a.out, "w"), indent=1)

    print("\n  VERDICT")
    ok = [f for f, v in res.items()
          if f != "0%" and v["pesq"] >= base["pesq"] - 0.01
          and v["stoi"] >= base["stoi"] - 0.005]
    if ok:
        print("    Quality survives at: %s" % ", ".join(ok))
        print("    -> redundancy exists; rebuilding a physically smaller model")
        print("       is justified. Masking saves no compute, removing does.")
    else:
        print("    Quality does NOT survive even the smallest masking tested.")
        print("    -> no redundancy to remove. Pruning is rejected for this")
        print("       model, and the reason is size: 22,956 parameters is")
        print("       already half of GTCRN's 48.2 k. The embedded bottleneck")
        print("       is operator support and sequential GRU steps, not")
        print("       parameter count, and pruning addresses neither.")
    print("\n  -> %s" % a.out)
