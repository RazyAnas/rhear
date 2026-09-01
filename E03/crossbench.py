#!/usr/bin/env python3
"""Is our MODEL behind, or is our TEST SET brutal? One experiment, no training.

GTCRN (Rong et al., ICASSP 2024) is 23.7k parameters -- essentially our size --
and reports PESQ 2.87 / STOI 0.940 on VoiceBank+DEMAND, above the PS's 2.5
target. We are 24,975 parameters and score 1.64 on our own defence set.

Two explanations, and they demand opposite responses:

  our model is behind    a same-size published model would also score ~2.8 on
                         our data -> our training or architecture is deficient
                         and their open recipe tells us what to copy.
  our test set is harder that same model scores ~1.6 on our data too -> our
                         architecture is competitive and the gap is the
                         benchmark, which changes what we claim, not what we build.

So run THEIR pretrained weights, through THEIR published inference path
(16 kHz, n_fft 512, hop 256, sqrt-Hann), over OUR held-out defence clips, scored
by OUR scorer so nothing differs but the model. Both their checkpoints are
included: DNS3 and VCTK-DEMAND.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))
GT = "/private/tmp/claude-501/-Users-mariyafatima-projectonboard/9740b1c3-cbf2-43bf-805d-45bfb359fbf4/scratchpad/gtcrn"
sys.path.insert(0, GT)

from train_interim import Pairs, DEV
from rhear_data.manifest import read_manifest
from eval_stratified import score_one, agg, SNR_BUCKETS


def gtcrn_enhance(model, wav, win):
    """Their published inference path, unmodified."""
    x = torch.from_numpy(wav.astype(np.float32))
    spec = torch.stft(x, 512, 256, 512, win, return_complex=False)
    with torch.no_grad():
        out = model(spec[None])[0]
    # newer torch requires a complex tensor here; their (F,T,2) real layout is
    # the same data, so view it as complex rather than changing the pipeline
    return torch.istft(torch.view_as_complex(out.contiguous()),
                       512, 256, 512, win).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "handoff", "data", "edef"))
    ap.add_argument("--ckpt", default="vctk", choices=["vctk", "dns3"])
    ap.add_argument("--out", default="runs/g012/crossbench")
    a = ap.parse_args()

    from gtcrn import GTCRN as TheirGTCRN
    model = TheirGTCRN().eval()
    f = {"vctk": "model_trained_on_vctk.tar", "dns3": "model_trained_on_dns3.tar"}[a.ckpt]
    ck = torch.load(os.path.join(GT, "checkpoints", f), map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    n_par = sum(p.numel() for p in model.parameters())
    print("  GTCRN pretrained on %s | %s params (ours: 24,975)" % (a.ckpt, f"{n_par:,}"))

    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    ds = Pairs(a.data, "test", rows, cache=False)
    meta = [r for r in rows if r["split"] == "test"]
    win = torch.hann_window(512).pow(0.5)
    print("  %s | %d clips\n" % (os.path.basename(a.data.rstrip('/')), len(ds)))

    out = []
    for i in range(len(ds)):
        x, y = ds[i]
        c, nz = y.numpy(), x.numpy()
        e = gtcrn_enhance(model, nz, win)
        L = min(len(c), len(nz), len(e))
        r = dict(meta[i]); r.update(score_one(c[:L], nz[:L], e[:L]))
        out.append(r)
        if (i + 1) % 60 == 0:
            print("    %d/%d" % (i + 1, len(ds)), flush=True)

    o = agg(out)
    print("\n  GTCRN (%s) on our defence set, %d clips" % (a.ckpt, o["n"]))
    print("    STOI %.4f -> %.4f    PESQ %.4f -> %.4f" %
          (o["stoi_n"], o["stoi_e"], o["pesq_n"], o["pesq_e"]))
    print("    SI-SDR %+.2f dB   SI-SIR %+.2f   SI-SAR %+.2f"
          % (o["sisdr_e"], o["sir_e"], o["sar_e"]))
    print("\n  by input SNR:")
    print("    %-12s%6s%9s%9s" % ("bucket", "n", "STOI", "PESQ"))
    res = {"overall": o, "by_snr": {}}
    for lo, hi, lab in SNR_BUCKETS:
        sel = [r for r in out if lo <= r.get("snr_db", -999) < hi]
        b = agg(sel); res["by_snr"][lab] = b
        if b:
            print("    %-12s%6d%9.3f%9.3f" % (lab, b["n"], b["stoi_e"], b["pesq_e"]))
    os.makedirs(os.path.join(HERE, a.out), exist_ok=True)
    tag = "%s_%s" % (a.ckpt, os.path.basename(a.data.rstrip('/')))
    json.dump(res, open(os.path.join(HERE, a.out, "gtcrn_%s.json" % tag), "w"), indent=1)
    print("\n  -> %s/gtcrn_%s.json" % (a.out, tag))
