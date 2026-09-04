#!/usr/bin/env python3
"""Fine-tune G7-base on VoiceBank+DEMAND.

Why this run exists. PS 26052 says, verbatim: "targeting SNR > 15 dB,
STOI > 0.85, and PESQ > 2.5" -- no input-SNR condition, no test set named. Those
thresholds match the conventions of the VoiceBank+DEMAND benchmark, on which
G7-base already scores STOI 0.931 and output SI-SDR 17.71 dB ZERO-SHOT, having
never seen the corpus, and misses PESQ by 0.146 (2.354 vs 2.5).

Every published number on that benchmark comes from a model trained on its own
training set. This is that run, and nothing more: same architecture, same loss,
same hop, initialised from G7-base.

Honesty conditions this run is bound by, because the number is only worth having
if it means what it says:
  · checkpoint selection is on HELD-OUT TRAIN SPEAKERS (p286, p287), never the
    test set. The 824-utterance test set is scored once, afterwards.
  · the result is a VoiceBank+DEMAND result. It does not transfer to our
    defence sets and must never be reported as if it did -- G7-base scores 1.635
    PESQ there, and so do two published systems (see 06-L1-artefact-evidence.md).

    /opt/anaconda3/bin/python E03/finetune_vbdemand.py --epochs 12
"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import N_FFT
from train_interim import enhance, stft, loss_fn, si_sdr, DEV
from eval_stratified import load_model

import soundfile as sf

FS = 16000
CROP = 4 * FS


class VB(Dataset):
    """4-second crops. Short utterances are zero-padded; VoiceBank utterances
    are 2-8 s so most are cropped rather than padded."""

    def __init__(self, root, rows, train=True, seed=0):
        self.root, self.rows, self.train = root, rows, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        c, _ = sf.read(os.path.join(self.root, r["clean"]), dtype="float32")
        z, _ = sf.read(os.path.join(self.root, r["noisy"]), dtype="float32")
        L = min(len(c), len(z))
        c, z = c[:L], z[:L]
        if L >= CROP:
            s = self.rng.integers(0, L - CROP + 1) if self.train else (L - CROP) // 2
            c, z = c[s:s + CROP], z[s:s + CROP]
        else:
            pad = CROP - L
            c = np.pad(c, (0, pad))
            z = np.pad(z, (0, pad))
        return torch.from_numpy(z), torch.from_numpy(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/vbd/prep16k")
    ap.add_argument("--init", default=os.path.join(HERE, "runs", "g7_hop256_50k", "best.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "g7_vbdemand"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.data, "manifest.json")))
    tr = VB(a.data, man["rows"]["train"], True, 1)
    va = VB(a.data, man["rows"]["val"], False, 2)
    print(f"\ntrain {len(tr)} | val {len(va)} (speakers {man['val_speakers']})")

    dl_tr = DataLoader(tr, batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model, _ = load_model(a.init)
    model = model.to(DEV)
    hop = int(model.hop_.item())
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"init  {a.init}\nmodel {n:,} params, hop {hop}, device {DEV}\n")

    win = torch.hann_window(N_FFT, device=DEV)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "max", factor=0.5, patience=1)

    best, bad, hist = -1e9, 0, []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, nb, skipped = time.time(), 0.0, 0, 0
        for z, c in dl_tr:
            z, c = z.to(DEV), c.to(DEV)
            opt.zero_grad()
            S_est, wav, X, spp = enhance(model, z, win)
            S_ref = stft(c, win, hop)
            l, _ = loss_fn(wav, c, S_est, S_ref, spp)
            if not torch.isfinite(l):
                skipped += 1
                continue
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(l); nb += 1

        model.eval()
        vs = []
        with torch.no_grad():
            for z, c in dl_va:
                z, c = z.to(DEV), c.to(DEV)
                _, wav, _, _ = enhance(model, z, win)
                vs.append(float(si_sdr(wav, c).mean()))
        v = float(np.mean(vs))
        sched.step(v)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1),
                     "val_sisdr": v, "skipped": skipped,
                     "lr": opt.param_groups[0]["lr"],
                     "secs": time.time() - t0})
        flag = ""
        if v > best:
            best, bad = v, 0
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
            flag = "  <- best"
        else:
            bad += 1
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
        print(f"  ep {ep:2d}/{a.epochs}  loss {tot / max(nb,1):8.4f}  "
              f"val SI-SDR {v:6.2f} dB  lr {opt.param_groups[0]['lr']:.1e}  "
              f"{time.time() - t0:5.0f}s{flag}"
              + (f"  [{skipped} non-finite batches]" if skipped else ""))
        if bad >= a.patience:
            print(f"\n  early stop: {a.patience} epochs without improvement")
            break

    print(f"\nbest val SI-SDR {best:.2f} dB")
    print(f"checkpoint {a.out}/best.pt")
    print(f"\nNow score the test set ONCE:")
    print(f"  /opt/anaconda3/bin/python E03/eval_vbdemand.py "
          f"--ckpt {a.out}/best.pt --root /tmp/vbd")


if __name__ == "__main__":
    main()
