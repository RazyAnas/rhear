#!/usr/bin/env python3
"""Mask-sharpening (alpha) sweep on a real single-mic recording. See docs/08 section 8.1."""
import numpy as np, sys, os, torch, soundfile as sf, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'/Users/mariyafatima/PS#2/E03'); sys.path.insert(0,'/Users/mariyafatima/PS#2/E03/model')
from eval_stratified import load_model
from train_interim import stft as tstft, istft as tistft, DEV, apply_deep_filter
from gtcrn_lite import N_FFT

OUTD='/Users/mariyafatima/PS#2/E03/runs/nearfield/real'; os.makedirs(OUTD,exist_ok=True)
model,_ = load_model('runs/g12_noise232/best.pt'); model.eval()
win = torch.hann_window(N_FFT, device=DEV)
hop = int(getattr(model,'hop_',torch.tensor(256)).item())

def run(x, alpha=1.0, floor_db=-40.0):
    """L1 with a sharpened mask: m -> m**alpha, floored so it cannot go silent."""
    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32)).to(DEV)[None]
        X = tstft(t, win, hop)
        spec = torch.stack([X.real, X.imag],1).transpose(2,3)
        mm, mp, spp, dfc = model(spec)
        mm = mm.clamp(min=0.0)**alpha
        mm = torch.clamp(mm, min=10**(floor_db/20))
        mag = X.abs().transpose(1,2).unsqueeze(1); ph = X.angle().transpose(1,2).unsqueeze(1)
        if mp is not None: ph = ph + mp
        Se = ((mm*mag)*torch.exp(1j*ph)).squeeze(1).transpose(1,2)
        if dfc is not None: Se = apply_deep_filter(Se, dfc, X, model.df_taps, model.df_bins)
        return tistft(Se, win, t.shape[-1], hop)[0].cpu().numpy()[:len(x)]

x,sr = sf.read('/Users/mariyafatima/Downloads/all_in.wav'); x=x.astype(np.float64)

def frames(y, n=320):
    m=len(y)//n; return (y[:m*n]**2).reshape(m,n).mean(1)
e0 = frames(x); q_lo, q_hi = np.quantile(e0,0.10), np.quantile(e0,0.70)
quiet = e0<=q_lo; loud = e0>=q_hi          # frame classes fixed from the INPUT

def stats(y):
    e=frames(y); k=min(len(e),len(quiet))
    nf=10*np.log10(e[:k][quiet[:k]].mean()+1e-20)
    sp=10*np.log10(e[:k][loud[:k]].mean()+1e-20)
    return nf, sp

nf0, sp0 = stats(x)
print(f"{'setting':22}{'noise floor':>13}{'speech':>9}{'noise cut':>11}{'speech cut':>12}{'net SNR':>9}")
print(f"{'INPUT (raw mic)':22}{nf0:>12.1f}{sp0:>9.1f}{'':>11}{'':>12}{'':>9}")
outs={'0_INPUT':x}
for a in (1.0,1.3,1.6,2.0,2.5):
    y = run(x, alpha=a); nf,sp = stats(y)
    print(f"{'L1  alpha=%.1f'%a:22}{nf:>12.1f}{sp:>9.1f}{nf0-nf:>+10.1f} dB{sp0-sp:>+11.1f} dB{(nf0-nf)-(sp0-sp):>+8.1f}")
    outs[f'L1_alpha{a:.1f}'.replace('.','p')] = y

pk = np.abs(x).max()
for k,y in outs.items():
    w=f'{OUTD}/{k}.wav'; sf.write(w,(y/pk*0.89).astype(np.float32),sr)
    os.system(f'ffmpeg -y -loglevel error -i "{w}" -codec:a libmp3lame -q:a 2 "{OUTD}/{k}.mp3"')
print("\nwritten", OUTD)
