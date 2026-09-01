#!/usr/bin/env python3
"""Is the ~10 dB SI-SAR a training failure or a limit of the 48-band grid?

H9 measured H3 at SI-SIR ~20 dB but SI-SAR ~10 dB, so SI-SDR is capped by
artefacts, not by leftover noise. Two very different worlds explain that:

  training problem       an oracle mask on OUR grid reaches high SI-SAR, so
                         48 bands can be artefact-free and the model simply
                         is not finding it -> fix the loss.
  representation problem an oracle on our grid is ALSO stuck near 10 dB, so
                         the band grid itself forces the artefacts -> no loss
                         change can help and the architecture must move.

Oracle families, all using the ideal ratio mask CLIPPED to <= 1, which is what
tanh(|m|) allows the real model:

  A  noisy, unprocessed
  B  ideal mask at 48 bands, NOISY phase      -- a magnitude-only masker
  C  ideal mask at 48 bands, CLEAN phase      -- OUR family's true ceiling,
                                                 since the model predicts phase
  D  ideal mask at 257 bins, CLEAN phase      -- full-resolution ceiling
  E  H3 as actually trained                   -- the gap to close

C is the number that decides it.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))
from gtcrn_lite import ERBSplit, N_FFT, HOP, FS, GTCRNLite
from train_interim import Pairs, enhance, stft, istft, DEV
from rhear_data.manifest import read_manifest
from phase5_comb import si_metrics, score


_ERB = {}


def band_limit(mask, nb):
    """Spread a per-bin mask onto nb ERB bands and back, energy-normalised."""
    key = (nb, str(mask.device))
    if key not in _ERB:                       # built once, not per clip
        _ERB[key] = ERBSplit(n_bands=nb).to(mask.device)
    erb = _ERB[key]
    mm = mask.transpose(1, 2).unsqueeze(1)                 # (B,1,T,F)
    mm = erb.inverse(erb(mm))
    mm = mm / (erb.inverse(erb(torch.ones_like(mm))) + 1e-8)
    return mm.squeeze(1).transpose(1, 2)


def synth(mag, phase, n):
    return istft((mag * torch.exp(1j * phase))[None].to(DEV),
                 torch.hann_window(N_FFT, device=DEV), n)[0].cpu().numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="runs/h9_frozen/h3_snapshot.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="runs/h9_frozen/sar_ceiling.json")
    a = ap.parse_args()

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    # Detect the variant from the checkpoint itself, the same way
    # eval_stratified.py does -- G0/G1 are phase-free, G2 adds the full-band
    # branch, and loading any of them into the default architecture fails.
    sd = torch.load(a.ckpt, map_location=DEV)
    k = "dec.3.pw.weight"
    has_phase = sd[k].shape[0] == 2 if k in sd else True
    has_fb = any(q.startswith("fb.") for q in sd)
    model = GTCRNLite(phase=has_phase, fullband=has_fb).to(DEV)
    model.load_state_dict(sd)
    model.eval()
    win = torch.hann_window(N_FFT, device=DEV)
    idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
    print("  %d clips from %s" % (len(idx), a.data))

    names = ["A noisy", "B ideal@48, noisy phase", "C ideal@48, CLEAN phase",
             "D ideal@257, CLEAN phase", "E H3 as trained"]
    acc = {k: {m: [] for m in ("stoi", "pesq", "sdr", "sir", "sar")}
           for k in names}

    for j, i in enumerate(idx):
        x, y = ds[int(i)]
        c = y.numpy().astype(np.float64)
        nz = x.numpy().astype(np.float64)
        noise = nz - c
        L = x.shape[-1]
        X = stft(x[None].to(DEV), win)
        S = stft(y[None].to(DEV), win)
        ideal = (S.abs() / (X.abs() + 1e-8)).clamp(0, 1)   # what tanh allows
        ph_n, ph_c = X.angle(), S.angle()

        outs = {
            "A noisy": nz,
            "B ideal@48, noisy phase":
                synth(band_limit(ideal, 48)[0] * X.abs()[0], ph_n[0], L),
            "C ideal@48, CLEAN phase":
                synth(band_limit(ideal, 48)[0] * X.abs()[0], ph_c[0], L),
            "D ideal@257, CLEAN phase":
                synth(ideal[0] * X.abs()[0], ph_c[0], L),
        }
        with torch.no_grad():
            _, est, _, _ = enhance(model, x[None].to(DEV), win)
        outs["E H3 as trained"] = est[0].cpu().numpy()

        for k, sig in outs.items():
            sig = np.asarray(sig, dtype=np.float64)
            M = min(len(sig), len(c))
            st, pq = score(c[:M], sig[:M])
            sd, si, sa = si_metrics(sig[:M], c[:M], noise[:M])
            for nm, v in zip(("stoi", "pesq", "sdr", "sir", "sar"),
                             (st, pq, sd, si, sa)):
                acc[k][nm].append(v)
        if (j + 1) % 20 == 0:
            print("    %d/%d" % (j + 1, len(idx)), flush=True)

    out = {}
    print("\n  %-26s%8s%8s%9s%9s%9s"
          % ("family", "STOI", "PESQ", "SI-SDR", "SI-SIR", "SI-SAR"))
    print("  " + "-" * 69)
    for k in names:
        v = {m: float(np.nanmean(acc[k][m])) for m in acc[k]}
        out[k] = v
        print("  %-26s%8.3f%8.3f%9.2f%9.2f%9.2f"
              % (k, v["stoi"], v["pesq"], v["sdr"], v["sir"], v["sar"]))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    c48, h3 = out["C ideal@48, CLEAN phase"], out["E H3 as trained"]
    print("\n  VERDICT")
    print("    SI-SAR ceiling on our own grid : %+.2f dB" % c48["sar"])
    print("    SI-SAR H3 actually achieves    : %+.2f dB" % h3["sar"])
    gap = c48["sar"] - h3["sar"]
    print("    headroom                       : %+.2f dB" % gap)
    print("    -> %s" % ("TRAINING problem: the grid allows far better SAR, "
                         "the model is not finding it" if gap > 4 else
                         "REPRESENTATION problem: the 48-band grid itself "
                         "forces these artefacts"))
    print("    -> %s" % (a.out))
