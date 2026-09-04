"""The chosen plan's variant: 96 ERB bands + a per-bin phase head below 5.7 kHz.

Two changes from GTCRNLite, both forced by measurement rather than taste
(docs/06-L1-artefact-evidence.md):

1. 96 bands instead of 48. A perfect mask on 48 bands caps PESQ at 2.392 at
   0 dB against a 2.5 target -- unreachable however training goes.

   Going to 96 bands would leave the encoder 12 wide, and the full-band branch
   lands 257 on exactly 6, so its assert fires. The fix is to take the ENCODER's
   last stride from 1 to 2: 96 -> 48 -> 24 -> 12 -> 6. The encoder width is 6
   again, the full-band branch is untouched, the DPRNN sees exactly the shape it
   sees today, and the decoder mirrors it with strides (2,2,2,2). No resize, no
   padding, no operator CMSIS-NN cannot run -- the same constraints that drove
   the original stride choices.

2. A per-bin phase head. Substituting clean phase below 4 kHz takes dSI-SDR at
   0 dB from 13.02 to 24.10 and PESQ from 2.392 to 4.261. No magnitude-only
   design reaches the dSI-SDR target with any margin; phase is the lever.

   The head reuses the deep-filter head's geometry -- a deconv chain from width
   6 to 183 bins (5.7 kHz), which covers the 4 kHz the ceiling analysis asked
   for -- and emits PER BIN. That is the whole difference from the phase branch
   this project removed: gtcrn_lite.py:124 records that it emitted one rotation
   per ERB band and spread it over bins up to 19 wide, "ill-posed in the wide
   ones". Below 4 kHz at 96 bands the widest band is 5 bins.

   It emits (cos, sin), not an angle, so there is no atan2 anywhere -- the other
   reason the original branch was removed, since CMSIS-NN cannot run it. The
   rotation is applied by complex multiply.

   RESIDUAL AROUND IDENTITY: the head outputs a delta and 1 is added to the
   cosine channel, so a freshly initialised head is the identity rotation. The
   variant therefore starts as a near-copy of G7-base and is fine-tuned, not
   retrained -- the same trick that made the deep-filter stage trainable.
"""

import torch
import torch.nn as nn

from gtcrn_lite import GTCRNLite, CausalConvBlock, CausalDeconvBlock, N_FFT, FS

PHASE_BINS = 183                 # what strides (4,4,2) with k=3 give from 6
PHASE_HZ = PHASE_BINS * FS / N_FFT


class PhaseHead(nn.Module):
    def __init__(self, c3):
        super().__init__()
        self.dec = nn.ModuleList([
            CausalDeconvBlock(c3, 24, stride=(1, 4)),
            CausalDeconvBlock(24, 16, stride=(1, 4)),
            CausalDeconvBlock(16, 2, stride=(1, 2), last=True),
        ])

    def forward(self, x):
        for i, d in enumerate(self.dec):
            x = d(x, (23, 91, PHASE_BINS)[i])
        cos = x[:, :1] + 1.0                       # identity at init
        sin = x[:, 1:]
        n = torch.sqrt(cos ** 2 + sin ** 2 + 1e-9)
        return cos / n, sin / n                    # unit vector, per bin


class GTCRNPhase(GTCRNLite):
    def __init__(self, ch=(32, 48, 48, 64), n_bands=96, hop=256, **kw):
        super().__init__(ch=ch, n_bands=n_bands, phase=False,
                         fullband=True, df=False, hop=hop, **kw)
        c0, c1, c2, c3 = ch
        self.enc = nn.ModuleList([
            CausalConvBlock(3, c0, stride=(1, 2), groups=1),
            CausalConvBlock(c0, c1, stride=(1, 2)),
            CausalConvBlock(c1, c2, stride=(1, 2)),
            CausalConvBlock(c2, c3, stride=(1, 2)),        # was (1,1)
        ])
        self.dec = nn.ModuleList([
            CausalDeconvBlock(c3 * 2, c2, stride=(1, 2)),  # was (1,1)
            CausalDeconvBlock(c2 * 2, c1, stride=(1, 2)),
            CausalDeconvBlock(c1 * 2, c0, stride=(1, 2)),
            CausalDeconvBlock(c0 * 2, 1, stride=(1, 2), last=True),
        ])
        self.phase_head = PhaseHead(c3)

    def forward(self, spec):
        """Returns (mask, (cos, sin), spp). The rotation is a UNIT VECTOR per
        bin for bins < PHASE_BINS, to be applied by complex multiply."""
        mag = torch.sqrt(spec[:, :1] ** 2 + spec[:, 1:] ** 2 + 1e-9)
        x0 = torch.cat([spec, mag], 1)
        x = self.erb(x0)
        skips, fdims = [], []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            fdims.append(x.shape[3])
        z = x0
        for b in self.fb:
            z = b(z)
        assert z.shape[3] == x.shape[3], (
            "full-band branch must land on the encoder's frequency width "
            "without a resize: got %d vs %d" % (z.shape[3], x.shape[3]))
        x = self.fuse(torch.cat([x, z], 1))
        spp = torch.sigmoid(self.spp(x))
        x = self.dprnn(x)
        cos, sin = self.phase_head(x)
        out_f = fdims[-2:-1] + fdims[-3:-2] + fdims[-4:-3] + [self.erb.n_bands]
        for d, sk, of in zip(self.dec, reversed(skips), out_f):
            x = d(torch.cat([x, sk], 1), of)
        m = self.erb.inverse(x)
        return torch.tanh(torch.abs(m)), (cos, sin), spp


def apply_phase(X, mask, cos, sin):
    """Se = |X|*mask * unit(X) * (cos + j sin), rotation only below PHASE_BINS.

    No atan2: the rotation is applied as a complex multiply, which is what the
    deployed graph would do too.
    """
    mag = X.abs().transpose(1, 2).unsqueeze(1)             # (B,1,T,F)
    unit = X.transpose(1, 2).unsqueeze(1) / (mag + 1e-9)   # (B,1,T,F) complex
    rot = torch.ones_like(unit)
    r = torch.complex(cos, sin).to(unit.dtype)             # (B,1,T,PHASE_BINS)
    rot[..., :PHASE_BINS] = r
    Se = (mask * mag) * unit * rot
    return Se.squeeze(1).transpose(1, 2)                   # (B,F,T)


def enhance_phase(model, x, win):
    """Mirrors train_interim.enhance for this variant. Returns (S_est, wav, X)."""
    from train_interim import stft, istft
    hop = int(model.hop_.item())
    X = stft(x, win, hop)
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
    mask, (cos, sin), spp = model(spec)
    Se = apply_phase(X, mask, cos, sin)
    return Se, istft(Se, win, x.shape[-1], hop), X, spp
