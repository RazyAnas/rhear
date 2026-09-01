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

def _pick_device():
    """CUDA first: this was MPS-only, so on an NVIDIA box it silently fell
    back to CPU and trained ~50x slower without saying so."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEV = os.environ.get("RHEAR_DEVICE") or _pick_device()


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
        """cache: True = float32 in RAM, "int16" = int16 in RAM, False = off.

        int16 exists for the 20k-mixture runs. float32 needs 10.2 GB there,
        past this machine's 16 GB; int16 needs 5.1 GB and is exact, because
        every mixture is written as PCM_16 either way. Streaming from disk
        instead is not an option: libsndfile is not fork-safe, so DataLoader
        workers die with an unformattable LibsndfileError, and a single
        worker makes the run disk-bound.
        """
        self.root = root
        self.rows = [r for r in manifest_rows if r["split"] == split]
        self.cache = None
        self.int16 = (cache == "int16")
        if cache and self.rows:
            dt = "int16" if self.int16 else "float32"

            def load(key):
                from concurrent.futures import ThreadPoolExecutor
                paths = [os.path.join(root, r[key]) for r in self.rows]
                with ThreadPoolExecutor(max_workers=8) as ex:
                    return np.stack(list(ex.map(
                        lambda p: sf.read(p, dtype=dt)[0], paths)))

            self.cache = (torch.from_numpy(load("noisy")),
                          torch.from_numpy(load("clean")))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.cache is not None:
            x, y = self.cache[0][i], self.cache[1][i]
            if self.int16:
                return x.float() / 32768.0, y.float() / 32768.0
            return x, y
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


def loss_fn(est_wav, ref_wav, S_est, S_ref, spp, c=0.3, rho=8.0, eps=1e-4):
    """SI-SDR + power-law-compressed ASYMMETRIC magnitude + RI + SPP.

    The asymmetry must live INSIDE the magnitude loss, not beside it. A first
    version used a symmetric magnitude term at weight 30 plus a separate
    asymmetric term at weight 8 with rho=6, which gives an effective
    remove-speech : leave-noise ratio of only 2.05x instead of 6x -- the loss was
    very nearly symmetric, the model over-suppressed, and STOI went DOWN (-0.008)
    while SI-SDR climbed. Folding rho into the one magnitude term makes the
    effective ratio exactly rho.
    """
    # NUMERICAL STABILITY (measured, not guessed).
    # The derivative of power-law compression |S|^c goes as |S|^(c-1) = |S|^-0.7,
    # so it diverges as a bin approaches zero. Early in training the mask sits
    # near 0.5 everywhere and nothing is near zero; LATE in training the mask
    # becomes bimodal (measured p25 = 0.003) and many bins do reach zero. That is
    # why 27.6% of batches went non-finite in the previous run and why none did
    # in epoch 1.
    #
    # Two fixes, both numerical -- the objective is unchanged where it matters:
    #   eps 1e-8 -> 1e-4 bounds the gradient by eps^(c-1): max |grad| 259.7 -> 10.1,
    #     for a loss change of 3.4e-3 on anything above -60 dBFS.
    #   S * (|S|+eps)^(c-1) replaces (|S|+eps)^c * exp(j*angle(S)). Algebraically
    #     identical (max |A-B| = 2.6e-7) but avoids angle(), whose gradient was a
    #     further 3.5x larger near zero.
    l_sisdr = -si_sdr(est_wav, ref_wav).mean()
    me = (S_est.abs() + eps) ** c
    mr = (S_ref.abs() + eps) ** c
    d = mr - me                                        # >0 = speech was removed
    l_mag = (rho * torch.clamp(d, min=0).pow(2).mean()
             + torch.clamp(-d, min=0).pow(2).mean())
    ce = S_est * (S_est.abs() + eps) ** (c - 1)
    cr = S_ref * (S_ref.abs() + eps) ** (c - 1)
    ri = ce - cr
    l_ri = (ri.real.pow(2) + ri.imag.pow(2)).mean()
    l_asym = torch.zeros((), device=me.device)
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
    total = l_sisdr + 30.0 * l_mag + 15.0 * l_ri + 0.5 * l_spp
    return total, dict(sisdr=float(l_sisdr), mag=float(l_mag))


def enhance(model, x, win):
    X = stft(x, win)
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)      # (B,2,T,F)
    mm, mp, spp = model(spec)
    mag = X.abs().transpose(1, 2).unsqueeze(1)
    ph = X.angle().transpose(1, 2).unsqueeze(1)
    if mp is not None:
        ph = ph + mp
    Se = (mm * mag) * torch.exp(1j * ph)
    Se = Se.squeeze(1).transpose(1, 2)
    return Se, istft(Se, win, x.shape[-1]), X, spp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    # H3: 20k pairs cached as float32 is 10.2 GB, past this machine's 16 GB.
    # Both default to Condition B's behaviour so the baseline protocol is
    # bit-identical; only the H3 runs pass them.
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-int16", action="store_true")
    ap.add_argument("--no-phase", action="store_true",
                    help="drop the learned phase branch (measured to improve "
                         "STOI, PESQ and SI-SAR on both eval sets)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/interim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rhear_data.manifest import read_manifest
    hdr, rows = read_manifest(os.path.join(a.data, "manifest.jsonl"))
    cache = "int16" if a.cache_int16 else (not a.no_cache)
    tr = torch.utils.data.DataLoader(Pairs(a.data, "train", rows, cache=cache),
                                     batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True,
                                     persistent_workers=a.workers > 0)
    va = torch.utils.data.DataLoader(Pairs(a.data, "val", rows, cache=cache),
                                     batch_size=a.batch, num_workers=a.workers)
    print(f"  device {DEV}   train {len(tr.dataset)}   val {len(va.dataset)}"
          f"   (cache: {cache})")

    model = GTCRNLite(phase=not a.no_phase).to(DEV)
    p = count_params(model); macs, _ = count_macs_per_frame(model)
    print(f"  model {p:,} params, {macs*(FS/HOP)/1e6:.1f} MMAC/s")
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    win = torch.hann_window(N_FFT, device=DEV)

    best, hist, skipped = -1e9, [], 0
    ep_skips = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tl = 0; n = 0
        sk0 = skipped
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
        ep_skips.append(skipped - sk0)
        hist.append(dict(epoch=ep, train_loss=tl / max(n, 1), val_sisdr=v,
                         skipped=skipped - sk0))
        if v > best:
            best = v
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d}  loss {tl/max(n,1):8.3f}   val SI-SDR {v:6.2f} dB"
                  f"   best {best:6.2f}   skip {skipped-sk0:2d}/{len(tr)}"
                  f"   {(time.time()-t0)/60:5.1f} min")
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min, best val SI-SDR {best:.2f} dB")
    print(f"  skipped {skipped} batches as non-finite "
          f"({100*skipped/max(skipped+len(hist)*len(tr),1):.2f}% of steps)")
    print(f"  checkpoint -> {os.path.join(a.out,'best.pt')}")
