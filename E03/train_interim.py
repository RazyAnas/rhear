#!/usr/bin/env python3
"""Train the L1 baseline on the INTERIM dataset (real speech, synthesised noise).

Runs on Apple MPS. Any metric produced here is an INTERIM number, not G3:
the noise is synthesised because the real corpora could not download in time.
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from gtcrn_lite import GTCRNLite, N_FFT, HOP, FS, count_params, count_macs_per_frame

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def stft(x, win):
    return torch.stft(x, N_FFT, HOP, N_FFT, win, return_complex=True,
                      center=True, pad_mode="constant")


def istft(X, win, n):
    return torch.istft(X, N_FFT, HOP, N_FFT, win, center=True, length=n)


class Pairs(torch.utils.data.Dataset):
    """Holds the whole split in RAM.

    2.9 h of 16 kHz float32 pairs is ~1.3 GB, which fits comfortably. Re-reading
    2,000 wav pairs from disk every epoch made the run disk-bound at 1.5 min per
    epoch when the compute alone is a few seconds.
    """
    def __init__(self, root, split, manifest_rows, cache=True):
        self.root = root
        self.rows = [r for r in manifest_rows if r["split"] == split]
        self.cache = None
        if cache and self.rows:
            X = np.stack([sf.read(os.path.join(root, r["noisy"]),
                                  dtype="float32")[0] for r in self.rows])
            Y = np.stack([sf.read(os.path.join(root, r["clean"]),
                                  dtype="float32")[0] for r in self.rows])
            self.cache = (torch.from_numpy(X), torch.from_numpy(Y))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.cache is not None:
            return self.cache[0][i], self.cache[1][i]
        r = self.rows[i]
        x, _ = sf.read(os.path.join(self.root, r["noisy"]), dtype="float32")
        y, _ = sf.read(os.path.join(self.root, r["clean"]), dtype="float32")
        return torch.from_numpy(x), torch.from_numpy(y)


def si_sdr(est, ref, eps=1e-6):
    ref = ref - ref.mean(-1, keepdim=True)
    est = est - est.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    t = a * ref
    return 10 * torch.log10((t.pow(2).sum(-1) + eps) / ((est - t).pow(2).sum(-1) + eps))


def loss_fn(est_wav, ref_wav, S_est, S_ref, spp, c=0.3, rho=6.0):
    """SI-SDR + power-law-compressed magnitude/RI + the asymmetric
    intelligibility term (docs/02-architecture.md 7.2)."""
    l_sisdr = -si_sdr(est_wav, ref_wav).mean()
    me = (S_est.abs() + 1e-8) ** c
    mr = (S_ref.abs() + 1e-8) ** c
    l_mag = (me - mr).pow(2).mean()
    ri = (me * torch.exp(1j * S_est.angle()) - mr * torch.exp(1j * S_ref.angle()))
    l_ri = (ri.real.pow(2) + ri.imag.pow(2)).mean()
    d = mr - me                                        # >0 = speech removed
    l_asym = (rho * torch.clamp(d, min=0).pow(2).mean()
              + torch.clamp(-d, min=0).pow(2).mean())
    # Speech-presence target, reduced to the encoder's band resolution.
    # `mr` is (B,1,T,F); `spp` is (B,1,T,bands) with the SAME T (the encoder does
    # not stride in time). So only the frequency axis is reduced -- an earlier
    # version pooled to (bands,T), which silently transposed time and frequency.
    # adaptive_avg_pool2d is unimplemented on MPS for non-divisible sizes, so
    # interpolate is used instead.
    # mr is (B,F,T) from the STFT; spp is (B,1,T,bands) with the SAME T.
    q = (mr > (0.15 * mr.mean())).float().transpose(1, 2).unsqueeze(1)  # (B,1,T,F)
    qt = nn.functional.interpolate(q, size=spp.shape[-2:], mode="nearest")
    l_spp = nn.functional.binary_cross_entropy(
        spp.clamp(1e-6, 1 - 1e-6), qt.clamp(0, 1))
    total = l_sisdr + 30.0 * l_mag + 15.0 * l_ri + 8.0 * l_asym + 0.5 * l_spp
    return total, dict(sisdr=float(l_sisdr), mag=float(l_mag), asym=float(l_asym))


def enhance(model, x, win):
    X = stft(x, win)
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)      # (B,2,T,F)
    mm, mp, spp = model(spec)
    mag = X.abs().transpose(1, 2).unsqueeze(1)
    ph = X.angle().transpose(1, 2).unsqueeze(1)
    Se = (mm * mag) * torch.exp(1j * (ph + mp))
    Se = Se.squeeze(1).transpose(1, 2)
    return Se, istft(Se, win, x.shape[-1]), X, spp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/interim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rhear_data.manifest import read_manifest
    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    tr = torch.utils.data.DataLoader(Pairs(a.data, "train", rows), batch_size=a.batch,
                                     shuffle=True, num_workers=0, drop_last=True)
    va = torch.utils.data.DataLoader(Pairs(a.data, "val", rows), batch_size=a.batch)
    print(f"  device {DEV}   train {len(tr.dataset)}   val {len(va.dataset)}"
          f"   (cached in RAM)")

    model = GTCRNLite().to(DEV)
    p = count_params(model); macs, _ = count_macs_per_frame(model)
    print(f"  model {p:,} params, {macs*(FS/HOP)/1e6:.1f} MMAC/s")
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    win = torch.hann_window(N_FFT, device=DEV)

    best, hist, skipped = -1e9, [], 0
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tl = 0; n = 0
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            # A handful of mixtures have a near-silent clean target (the pair is
            # normalised together, so at -10 dB SNR the speech can end up ~50 dB
            # below full scale). SI-SDR is fragile there. Skip and COUNT rather
            # than letting one batch poison the run with a NaN.
            if not torch.isfinite(y).all() or float(y.pow(2).mean()) < 1e-9:
                skipped += 1
                continue
            Se, est, X, spp = enhance(model, x, win)
            Sr = stft(y, win)
            loss, parts = loss_fn(est, y, Se, Sr, spp)
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad()
                continue
            opt.zero_grad(); loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                skipped += 1
                opt.zero_grad()
                continue
            opt.step()
            tl += float(loss); n += 1
        sched.step()
        model.eval(); vs = []
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEV), y.to(DEV)
                _, est, _, _ = enhance(model, x, win)
                q = si_sdr(est, y)
                vs += [float(t) for t in q if torch.isfinite(t)]
        v = float(np.mean(vs)) if vs else float("nan")
        hist.append(dict(epoch=ep, train_loss=tl / max(n, 1), val_sisdr=v))
        if v > best:
            best = v
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d}  loss {tl/max(n,1):8.3f}   val SI-SDR {v:6.2f} dB"
                  f"   best {best:6.2f}   {(time.time()-t0)/60:5.1f} min")
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min, best val SI-SDR {best:.2f} dB")
    print(f"  skipped {skipped} batches as non-finite "
          f"({100*skipped/max(skipped+len(hist)*len(tr),1):.2f}% of steps)")
    print(f"  checkpoint -> {os.path.join(a.out,'best.pt')}")
