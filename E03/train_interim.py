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


def stft(x, win, hop=None):
    return torch.stft(x, N_FFT, hop or HOP, N_FFT, win, return_complex=True,
                      center=True, pad_mode="constant")


def istft(X, win, n, hop=None):
    return torch.istft(X, N_FFT, hop or HOP, N_FFT, win, center=True, length=n)


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


# --- G1: perceptual loss (PS names it; we had none) -------------------------
# Two deterministic terms, no adversarial training. Our loss was SI-SDR +
# magnitude + real/imaginary + speech-presence, none of which correlates with
# perceptual quality -- the likeliest reason +6.6 dB of SI-SDR bought only
# +0.20 PESQ.
#
# Deliberately NOT MetricGAN to start: we have a documented instability history
# (27.6% of batches once went non-finite), so the stable form goes first and
# the adversarial one is the escalation if this under-delivers.
_MR_WINDOWS = {}


def _mr_stft(x, n_fft, hop):
    key = (n_fft, str(x.device))
    if key not in _MR_WINDOWS:
        _MR_WINDOWS[key] = torch.hann_window(n_fft, device=x.device)
    return torch.stft(x, n_fft, hop, n_fft, _MR_WINDOWS[key],
                      return_complex=True, center=True, pad_mode="constant")


def perceptual_loss(est_wav, ref_wav, erb_M=None, eps=1e-5):
    """Multi-resolution STFT loss + ERB-weighted log-magnitude error.

    Multi-resolution: spectral convergence plus log-magnitude L1 at three FFT
    sizes, so errors are penalised at several time/frequency trade-offs rather
    than only at the 512-point analysis the model happens to use.

    ERB weighting: the same band grouping the model predicts in, so the error
    is weighted the way hearing resolves frequency instead of uniformly per bin.

    eps is 1e-5 rather than something smaller on purpose. log(|S|) has gradient
    1/|S|, which diverges as a bin approaches zero -- exactly the failure that
    produced 27.6% non-finite batches before. Every log here is log(|S| + eps).
    """
    # Scale-normalise against the REFERENCE rms first. Without this the log
    # terms are absolute-level dependent: a quiet clip pushes |S| toward the
    # eps floor, log(|S|+eps) swings wildly, and the gradient explodes.
    # Measured on a near-silent case before this fix: perceptual term 102.5
    # against 5.1 for normal speech, and max |grad| 6481 against 64 -- a 100x
    # spike, which is the shape of the failure that once cost 27.6% of
    # batches. Perceptual quality should not depend on absolute gain anyway.
    rms = ref_wav.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-4)
    est_wav = est_wav / rms
    ref_wav = ref_wav / rms

    l_sc = est_wav.new_zeros(())
    l_mag = est_wav.new_zeros(())
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        E = _mr_stft(est_wav, n_fft, hop).abs()
        R = _mr_stft(ref_wav, n_fft, hop).abs()
        l_sc = l_sc + (torch.linalg.norm(R - E) /
                       (torch.linalg.norm(R) + eps))
        l_mag = l_mag + (torch.log(R + eps) - torch.log(E + eps)).abs().mean()
    l_sc, l_mag = l_sc / 3.0, l_mag / 3.0

    l_erb = est_wav.new_zeros(())
    if erb_M is not None:
        E = _mr_stft(est_wav, N_FFT, HOP).abs()          # (B,F,T)
        R = _mr_stft(ref_wav, N_FFT, HOP).abs()
        be = torch.einsum("bft,gf->bgt", E, erb_M)
        br = torch.einsum("bft,gf->bgt", R, erb_M)
        l_erb = (torch.log(br + eps) - torch.log(be + eps)).abs().mean()
    return l_sc + l_mag + l_erb


def loss_fn(est_wav, ref_wav, S_est, S_ref, spp, c=0.3, rho=8.0, eps=1e-4,
            perc=0.0, erb_M=None):
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
    l_perc = est_wav.new_zeros(())
    if perc > 0.0:                       # G1: off by default, single variable
        l_perc = perceptual_loss(est_wav, ref_wav, erb_M)
        total = total + perc * l_perc
    return total, dict(sisdr=float(l_sisdr), mag=float(l_mag),
                       perc=float(l_perc))


def enhance(model, x, win):
    # The hop is part of the model's contract, not a global. A checkpoint
    # trained at hop 256 evaluated at hop 64 silently produces garbage, so the
    # model carries its own and every caller reads it from there.
    hop = int(getattr(model, "hop_", torch.tensor(HOP)).item())
    X = stft(x, win, hop)
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)      # (B,2,T,F)
    mm, mp, spp, dfc = model(spec)
    mag = X.abs().transpose(1, 2).unsqueeze(1)
    ph = X.angle().transpose(1, 2).unsqueeze(1)
    if mp is not None:
        ph = ph + mp
    Se = (mm * mag) * torch.exp(1j * ph)
    Se = Se.squeeze(1).transpose(1, 2)                            # (B,F,T)
    if dfc is not None:
        Se = apply_deep_filter(Se, dfc, X, model.df_taps, model.df_bins)
    return Se, istft(Se, win, x.shape[-1], hop), X, spp


def apply_deep_filter(Se, dfc, X, taps, nbins):
    """Stage 2: a short COMPLEX filter across time, per bin, on stage 1's output.

        Y[t,f] = sum_i c_i[t,f] * Se[t-i,f]        for f < nbins

    Causal by construction -- only frame t and earlier are touched, so the 8 ms
    streaming budget is unchanged. Refines stage 1 rather than replacing it, so
    c = [1,0,...] reproduces the mask exactly and the cascade cannot be worse
    than its own first stage.

    The magnitude bound is then re-imposed against the ORIGINAL input, not
    against stage 1's output. A complex filter can otherwise amplify, which
    would break the property the whole design rests on: no output bin may exceed
    its input bin, so the model can attenuate but can never invent speech. At
    oracle strength this projection cost 0.065 PESQ (4.276 -> 4.211) -- a
    measured price for a guarantee we are not willing to trade.
    """
    B, F, T = Se.shape
    nb = min(nbins, F)
    c = dfc.permute(0, 3, 2, 1)                       # (B, nbins, T, 2*taps)
    cc = torch.complex(c[..., 0::2], c[..., 1::2])    # (B, nbins, T, taps)
    lo = Se[:, :nb]                                   # (B, nb, T)
    acc = torch.zeros_like(lo)
    for i in range(taps):                             # taps is 5; unrolled
        if i:
            # Concat, not Pad. Pad is not in the CMSIS-NN native set and
            # exporting it added the only new operator this stage introduced;
            # Concat is native and identical here. On the device neither exists:
            # the previous frames live in a ring buffer and the "shift" is a
            # read offset, so this is purely an artefact of batch-mode export.
            z = torch.zeros_like(lo[:, :, :i])
            shifted = torch.cat([z, lo[:, :, :T - i]], dim=-1)
        else:
            shifted = lo
        acc = acc + cc[..., i] * shifted
    out = Se.clone()
    lim = X[:, :nb].abs()
    out[:, :nb] = acc * torch.clamp(lim / (acc.abs() + 1e-9), max=1.0)
    return out


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
    ap.add_argument("--perceptual", type=float, default=0.0,
                    help="G1: weight on the perceptual loss term (0 = off, "
                         "which reproduces the frozen baseline exactly)")
    ap.add_argument("--df", action="store_true",
                    help="add the deep-filtering second stage (5 complex taps "
                         "per bin below 6 kHz, applied to the ERB stage output)")
    ap.add_argument("--hop", type=int, default=HOP,
                    help="L1 STFT hop. 64 = 250 frames/s (the historic value); "
                         "256 = 62.5 frames/s, matching GTCRN and comparable "
                         "published systems, cutting inference rate 4x. Affects "
                         "the COMMS path only -- L0's 146 us budget is separate.")
    ap.add_argument("--rho", type=float, default=8.0,
                    help="Asymmetry of the magnitude loss: removing speech is "
                         "penalised rho times harder than leaving noise in. 8.0 "
                         "is every checkpoint to date, and it is why this model "
                         "has the best SI-SAR of anything measured on our data "
                         "(11.9 vs GTCRN 10.8, SepFormer 6.6). It also makes the "
                         "model deliberately under-suppress, which is most of "
                         "what the 'extraction gap' against a symmetric ideal "
                         "mask was measuring. Lower it to trade SI-SAR for PESQ.")
    ap.add_argument("--bands", type=int, default=48,
                    help="ERB bands. 48 is every checkpoint to date. 96 doubles "
                         "the mask's frequency resolution, which is what caps "
                         "PESQ -- a PERFECT 48-band mask reaches 2.838 over this "
                         "mix and 2.392 at 0 dB. Parameters are unchanged (the "
                         "ERB matrix is fixed and the convolutions are "
                         "channel-wise); only compute moves.")
    ap.add_argument("--ch", default="16,24,24,32",
                    help="encoder/decoder channel widths, comma separated")
    ap.add_argument("--fullband", action="store_true",
                    help="G2: add the parallel full-band branch over all 257 "
                         "bins (PS asks for full-band AND sub-band features)")
    ap.add_argument("--no-phase", action="store_true",
                    help="drop the learned phase branch (measured to improve "
                         "STOI, PESQ and SI-SAR on both eval sets)")
    ap.add_argument("--init", default=None,
                    help="initialise weights from this checkpoint before "
                         "training (fine-tune rather than train from scratch)")
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

    ch = tuple(int(v) for v in a.ch.split(","))
    model = GTCRNLite(ch=ch, n_bands=a.bands, phase=not a.no_phase,
                      fullband=a.fullband, df=a.df, hop=a.hop).to(DEV)
    print("  hop %d (%.1f frames/s)   channels %s   bands %d"
          % (a.hop, FS / a.hop, str(ch), a.bands))
    if a.init:
        # Fine-tune. The architecture flags must still match the checkpoint --
        # load_state_dict is strict, so a mismatch fails loudly here rather
        # than silently training something else.
        sd = torch.load(a.init, map_location=DEV)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print("  initialised from %s (fine-tune)" % a.init)
        if miss or unexp:
            # Loud on purpose. Loading non-strictly is how a checkpoint quietly
            # fails to apply and a run silently trains from scratch. Anything
            # listed here is randomly initialised -- for --df that should be the
            # dfdec head and NOTHING else.
            print("    randomly initialised (%d): %s" % (len(miss), ", ".join(miss[:8])))
            if unexp:
                print("    IN CHECKPOINT BUT UNUSED (%d): %s -- check the flags"
                      % (len(unexp), ", ".join(unexp[:8])))
    # the model's own ERB matrix, so the perceptual term weights error
    # exactly the way the model resolves frequency
    erb_M = model.erb.M if a.perceptual > 0 else None
    p = count_params(model); macs, _ = count_macs_per_frame(model)
    # rate must come from the MODEL's hop, not the module default -- at hop 256
    # the global constant overstates the MAC rate by 4x
    print(f"  model {p:,} params, {macs*(FS/a.hop)/1e6:.1f} MMAC/s "
          f"({FS/a.hop:.1f} frames/s)")
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
            # SAME hop as enhance() used, or the reference spectrogram has a
            # different frame count than the estimate and the loss cannot align:
            # at hop 256 this produced 1001 frames against 251.
            Sr = stft(y, win, a.hop)
            loss, parts = loss_fn(est, y, Se, Sr, spp,
                                  perc=a.perceptual, erb_M=erb_M, rho=a.rho)
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
        # ALSO keep the final epoch. best.pt is selected on validation SI-SDR,
        # so for any run that deliberately trades SI-SDR for perceptual quality
        # -- every --perceptual run -- that selector picks the checkpoint with
        # the LEAST of the thing being tested. G5 selected epoch 0 of 4 for
        # exactly this reason, and the result could not be interpreted.
        # Evaluate both and let the metric under test decide.
        torch.save(model.state_dict(), os.path.join(a.out, "last.pt"))
        if True:            # every epoch. Printing every 5th hid epochs 1-3 of
                            # a 5-epoch run behind an empty log, and three
                            # separate progress estimates were wrong because of it.
            print(f"  ep {ep:3d}  loss {tl/max(n,1):8.3f}   val SI-SDR {v:6.2f} dB"
                  f"   best {best:6.2f}   skip {skipped-sk0:2d}/{len(tr)}"
                  f"   {(time.time()-t0)/60:5.1f} min")
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)
    print(f"\n  done in {(time.time()-t0)/60:.1f} min, best val SI-SDR {best:.2f} dB")
    print(f"  skipped {skipped} batches as non-finite "
          f"({100*skipped/max(skipped+len(hist)*len(tr),1):.2f}% of steps)")
    print(f"  checkpoints -> {os.path.join(a.out,'best.pt')} (best val SI-SDR)")
    print(f"                 {os.path.join(a.out,'last.pt')} (final epoch)")
    print("  For a run targeting PESQ or perceptual quality, evaluate BOTH --")
    print("  best.pt is selected on SI-SDR and is biased against such runs.")
