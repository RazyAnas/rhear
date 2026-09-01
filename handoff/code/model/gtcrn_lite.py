"""RHEAR L1 baseline: a causal, GTCRN-class speech enhancer.

Design constraints come from docs/02-architecture.md:
  * causal only -- no future context anywhere, so the streaming and offline
    outputs are identical
  * <= 100 k parameters, <= 60 MMAC/s   (GTCRN is 48.2 k / 33.0 MMAC/s)
  * 8 ms algorithmic latency via asymmetric analysis/synthesis windows
  * bounded complex mask, so the network cannot invent energy
  * multi-task: mask + speech-presence probability

Nothing here is trained yet. This module exists so the parameter count, MAC rate
and latency can be MEASURED rather than estimated.
"""
import math
import torch
import torch.nn as nn

FS = 16_000
N_FFT = 512            # 32 ms analysis -> frequency resolution
HOP = 64               # 4 ms
SYNTH = 64             # 4 ms synthesis window (asymmetric)
N_BINS = N_FFT // 2 + 1        # 257
ALG_LATENCY_MS = 1000.0 * (SYNTH + HOP) / FS      # 8.0 ms


class ERBSplit(nn.Module):
    """Fixed ERB-style band grouping: 257 linear bins -> n_bands.

    Not learned, so it costs no parameters and its MACs are a sparse matmul.
    """
    def __init__(self, n_bins=N_BINS, n_bands=64, fs=FS):
        super().__init__()
        hz = torch.linspace(0, fs / 2, n_bins)
        erb = 21.4 * torch.log10(1 + 0.00437 * hz)
        edges = torch.linspace(erb[0], erb[-1], n_bands + 1)
        M = torch.zeros(n_bands, n_bins)
        for b in range(n_bands):
            m = (erb >= edges[b]) & (erb <= edges[b + 1])
            if m.sum() == 0:
                m[torch.argmin((erb - edges[b]).abs())] = True
            M[b, m] = 1.0 / m.sum()
        self.register_buffer("M", M)
        self.register_buffer("Minv", (M > 0).float().T)
        self.n_bands, self.n_bins = n_bands, n_bins

    def forward(self, x):                      # (B,C,T,F) -> (B,C,T,bands)
        return torch.einsum("bctf,gf->bctg", x, self.M)

    def inverse(self, x):                      # (B,C,T,bands) -> (B,C,T,bins)
        return torch.einsum("bctg,fg->bctf", x, self.Minv)


class CausalConvBlock(nn.Module):
    """Depthwise-separable 2D conv, causal in time, with channel shuffle."""
    def __init__(self, cin, cout, k=(3, 3), stride=(1, 2), groups=2):
        super().__init__()
        groups = math.gcd(groups, math.gcd(cin, cout))   # grouping must divide both
        self.pad_t = k[0] - 1
        self.pad_f = (k[1] - 1) // 2
        self.dw = nn.Conv2d(cin, cin, k, stride=stride, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)
        self.groups = groups

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad_f, self.pad_f, self.pad_t, 0))
        x = self.pw(self.dw(x))
        b, c, t, f = x.shape
        if self.groups > 1:                    # channel shuffle
            x = x.view(b, self.groups, c // self.groups, t, f
                       ).transpose(1, 2).reshape(b, c, t, f)
        return self.act(self.bn(x))


class CausalDeconvBlock(nn.Module):
    """Transposed depthwise-separable block. The time dimension is trimmed from
    the FRONT so the block stays causal -- a transposed conv otherwise spreads
    each input frame into future output frames."""
    def __init__(self, cin, cout, k=(3, 3), stride=(1, 2), groups=2, last=False):
        super().__init__()
        g = math.gcd(groups, math.gcd(cin, cout))
        self.pad_t = k[0] - 1
        self.dw = nn.ConvTranspose2d(cin, cin, k, stride=stride, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, groups=1 if last else g, bias=False)
        self.bn = nn.Identity() if last else nn.BatchNorm2d(cout)
        self.act = nn.Identity() if last else nn.PReLU(cout)

    def forward(self, x, out_f):
        x = self.pw(self.dw(x))
        x = x[:, :, :x.shape[2] - self.pad_t, :]          # causal trim
        x = x[:, :, :, :out_f] if x.shape[3] >= out_f else \
            nn.functional.pad(x, (0, out_f - x.shape[3]))
        return self.act(self.bn(x))


class DualPathRNN(nn.Module):
    """Intra-frame (frequency, bidirectional -- allowed, it does not look ahead
    in TIME) and inter-frame (time, unidirectional -- must be causal)."""
    def __init__(self, ch, hid_f=24, hid_t=32):
        super().__init__()
        self.gru_f = nn.GRU(ch, hid_f, batch_first=True, bidirectional=True)
        self.lin_f = nn.Linear(2 * hid_f, ch)
        self.gru_t = nn.GRU(ch, hid_t, batch_first=True, bidirectional=False)
        self.lin_t = nn.Linear(hid_t, ch)
        self.n_f = nn.LayerNorm(ch)
        self.n_t = nn.LayerNorm(ch)

    def forward(self, x):                      # (B,C,T,F)
        b, c, t, f = x.shape
        y = x.permute(0, 2, 3, 1).reshape(b * t, f, c)
        y, _ = self.gru_f(y)
        y = self.n_f(self.lin_f(y)).reshape(b, t, f, c)
        x = x + y.permute(0, 3, 1, 2)
        y = x.permute(0, 3, 2, 1).reshape(b * f, t, c)
        y, _ = self.gru_t(y)
        y = self.n_t(self.lin_t(y)).reshape(b, f, t, c)
        return x + y.permute(0, 3, 2, 1)


class GTCRNLite(nn.Module):
    def __init__(self, ch=(16, 24, 24, 32), n_bands=48, phase=True):
        """phase=False removes the learned phase branch.

        Measured on the final H3 checkpoint over 600 clips (E_def + E_env),
        switching the branch off improves STOI +0.023/+0.022, PESQ
        +0.041/+0.038 and SI-SAR +1.62/+1.65 dB. The rotation is emitted at
        48 ERB bands and applied to every bin in a band, but phase is
        circular and wraps quickly across frequency while our bands reach
        19 bins wide -- one rotation per band is ill-posed in the wide ones.
        Removing it also drops Atan, which CMSIS-NN cannot run.

        Defaults to True so existing checkpoints load unchanged.
        """
        super().__init__()
        self.phase = phase
        self.erb = ERBSplit(n_bands=n_bands)
        c0, c1, c2, c3 = ch
        self.enc = nn.ModuleList([
            CausalConvBlock(3, c0, stride=(1, 2), groups=1),
            CausalConvBlock(c0, c1, stride=(1, 2)),
            CausalConvBlock(c1, c2, stride=(1, 2)),
            CausalConvBlock(c2, c3, stride=(1, 1)),
        ])
        self.dprnn = DualPathRNN(c3)
        self.dec = nn.ModuleList([
            CausalDeconvBlock(c3 * 2, c2, stride=(1, 1)),
            CausalDeconvBlock(c2 * 2, c1, stride=(1, 2)),
            CausalDeconvBlock(c1 * 2, c0, stride=(1, 2)),
            CausalDeconvBlock(c0 * 2, 2 if phase else 1,
                              stride=(1, 2), last=True),
        ])
        self.spp = nn.Conv2d(c3, 1, 1)          # speech-presence head (multi-task)

    def forward(self, spec):
        """spec: (B,2,T,F) real/imag. Returns bounded complex mask and SPP."""
        mag = torch.sqrt(spec[:, :1] ** 2 + spec[:, 1:] ** 2 + 1e-9)
        x = torch.cat([spec, mag], 1) ** 1.0
        x = self.erb(x)                          # -> (B,3,T,bands)
        skips, fdims = [], []
        for e in self.enc:
            x = e(x)
            skips.append(x); fdims.append(x.shape[3])
        spp = torch.sigmoid(self.spp(x))
        x = self.dprnn(x)
        out_f = fdims[-2:-1] + fdims[-3:-2] + fdims[-4:-3] + [self.erb.n_bands]
        for d, sk, of in zip(self.dec, reversed(skips), out_f):
            x = d(torch.cat([x, sk], 1), of)
        m = self.erb.inverse(x)                  # bands -> bins
        if not self.phase:
            # magnitude only; the noisy phase passes through untouched
            return torch.tanh(torch.abs(m)), None, spp
        mr, mi = m[:, :1], m[:, 1:]
        mmag = torch.tanh(torch.sqrt(mr ** 2 + mi ** 2 + 1e-9))   # bounded
        mph = torch.atan2(mi, mr + 1e-9)
        return mmag, mph, spp


# ------------------------------------------------------------------ analysis
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_macs_per_frame(model, n_bands=48):
    """Analytic MACs for ONE frame. Conv: Cin*Cout*K*F/groups. GRU: 3*(in+h)*h."""
    macs, detail = 0, {}
    f = n_bands
    for i, e in enumerate(model.enc):
        cin = e.dw.in_channels; cout = e.pw.out_channels
        kt, kf = e.dw.kernel_size; st = e.dw.stride[1]
        fo = math.ceil(f / st)
        dw = cin * kt * kf * fo
        pw = cin * cout * fo // e.groups
        macs += dw + pw; detail[f"enc{i}"] = dw + pw; f = fo
    c = model.dprnn.gru_f.input_size
    hf, ht = model.dprnn.gru_f.hidden_size, model.dprnn.gru_t.hidden_size
    gf = 2 * f * 3 * (c + hf) * hf + f * 2 * hf * c
    gt = f * 3 * (c + ht) * ht + f * ht * c
    macs += gf + gt; detail["dprnn_freq"] = gf; detail["dprnn_time"] = gt
    fo = f
    for i, d in enumerate(model.dec):
        cin = d.dw.in_channels; cout = d.pw.out_channels
        kt, kf = d.dw.kernel_size; st = d.dw.stride[1]
        fu = fo * st
        dw = cin * kt * kf * fu
        pw = cin * cout * fu
        macs += dw + pw; detail[f"dec{i}"] = dw + pw; fo = fu
    detail["spp"] = model.dprnn.gru_f.input_size * f
    macs += detail["spp"]
    stft = 2 * (5 * N_FFT * math.log2(N_FFT))
    detail["stft+istft"] = int(stft)
    macs += int(stft)
    return macs, detail


if __name__ == "__main__":
    m = GTCRNLite().eval()
    p = count_params(m)
    macs, detail = count_macs_per_frame(m)
    fps = FS / HOP
    with torch.no_grad():
        spec = torch.randn(1, 2, 25, N_BINS)
        mm, mp, spp = m(spec)
    print("RHEAR L1 baseline - GTCRNLite\n" + "=" * 62)
    print(f"  input / output      {FS} Hz mono")
    print(f"  n_fft {N_FFT} ({1000*N_FFT/FS:.0f} ms)   hop {HOP} ({1000*HOP/FS:.0f} ms)"
          f"   synthesis {SYNTH} ({1000*SYNTH/FS:.0f} ms)")
    print(f"  frame rate          {fps:.0f} fps")
    print(f"  algorithmic latency {ALG_LATENCY_MS:.1f} ms  (asymmetric windows:"
          f" synthesis + hop, not analysis + hop)")
    print(f"  parameters          {p:,}   ({p*1/1e3:.1f} k, INT8 ~= {p/1e3:.0f} KB)")
    print(f"  MACs per frame      {macs:,}")
    print(f"  MAC rate            {macs*fps/1e6:.1f} MMAC/s")
    print(f"  output shapes       mask {tuple(mm.shape)}  phase {tuple(mp.shape)}"
          f"  spp {tuple(spp.shape)}")
    print("\n  per-block MACs/frame:")
    for k, v in sorted(detail.items(), key=lambda x: -x[1]):
        print(f"    {k:<14}{v:10,}  ({100*v/macs:5.1f}%)")
    print("\n  budget check (docs/02-architecture.md 9.2/9.3):")
    print(f"    params  <= 100 k  : {'PASS' if p <= 100_000 else 'FAIL'} ({p/1e3:.1f} k)")
    print(f"    MMAC/s  <= 60     : {'PASS' if macs*fps/1e6 <= 60 else 'FAIL'}"
          f" ({macs*fps/1e6:.1f})")
    print(f"    latency <= 8 ms   : {'PASS' if ALG_LATENCY_MS <= 8.0 else 'FAIL'}"
          f" ({ALG_LATENCY_MS:.1f} ms)")
