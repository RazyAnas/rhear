#!/usr/bin/env python3
"""Final L1 chain: near-field ILD mask -> GTCRNLite (sharpened mask) -> near-field gate.
Produces the 8 s before/after demo in runs/nearfield/zero/. See docs/08.
"""
import numpy as np, sys, os, torch, soundfile as sf, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'/Users/mariyafatima/PS#2/E03'); sys.path.insert(0,'/Users/mariyafatima/PS#2/E03/model')
from nearfield_sim import load_speech, render, diffuse_pair, GEOM, stft2, istft2
from scipy.signal import resample_poly
from eval_stratified import load_model
from train_interim import stft as tstft, istft as tistft, DEV, apply_deep_filter
from gtcrn_lite import N_FFT
from pystoi import stoi as stoi_fn
from pesq import pesq as pesq_fn
FS16=16000; OUTD='/Users/mariyafatima/PS#2/E03/runs/nearfield/zero'; os.makedirs(OUTD,exist_ok=True)
model,_=load_model('runs/g12_noise232/best.pt'); model.eval()
win=torch.hann_window(N_FFT,device=DEV); hop=256

# ---- 0. does the 512/128 STFT round-trip preserve energy? -------------------
z=np.random.default_rng(0).standard_normal(128000)*0.1
rt=istft2(stft2(z,nfft=512,hop=128)[2],nfft=512,hop=128,n=len(z))
k=min(len(rt),len(z))
print(f"round-trip 512/128: err {10*np.log10(np.mean((rt[:k]-z[:k])**2)/np.mean(z[:k]**2)):.1f} dB"
      f" | energy ratio {10*np.log10(np.mean(rt[:k]**2)/np.mean(z[:k]**2)):+.2f} dB"
      f" | len {len(rt)} vs {len(z)}")

def L1(x, alpha=1.6, floor_db=-40.):
    with torch.no_grad():
        t=torch.from_numpy(x.astype(np.float32)).to(DEV)[None]
        X=tstft(t,win,hop); mm,mp,spp,dfc=model(torch.stack([X.real,X.imag],1).transpose(2,3))
        mm=torch.clamp(mm.clamp(min=0.)**alpha, min=10**(floor_db/20))
        mag=X.abs().transpose(1,2).unsqueeze(1); ph=X.angle().transpose(1,2).unsqueeze(1)
        if mp is not None: ph=ph+mp
        Se=((mm*mag)*torch.exp(1j*ph)).squeeze(1).transpose(1,2)
        if dfc is not None: Se=apply_deep_filter(Se,dfc,X,model.df_taps,model.df_bins)
        return tistft(Se,win,t.shape[-1],hop)[0].cpu().numpy()[:len(x)]

def nf_gate(y, il, M1, frame=128, pre=4, post=10, gate_db=-70., ild_thr=2.0):
    """Gate driven by the near-field cue: energy-weighted ILD per frame."""
    w=np.abs(M1)**2
    fi=(w*il).sum(0)/(w.sum(0)+1e-20)              # per-frame ILD, energy weighted
    op=fi>ild_thr
    op=np.convolve(op.astype(float),np.ones(pre+post+1),mode='same')>0
    ramp=np.convolve(op.astype(float),np.ones(5)/5,mode='same')
    g=10**(gate_db/20)+(1-10**(gate_db/20))*np.clip(ramp,0,1)
    gs=np.repeat(g,frame)[:len(y)]
    return y[:len(gs)]*gs

def squelch(y, fs=FS16, frame=256, pre=4, post=14, gate_db=-70., margin_db=10.):
    m=len(y)//frame; e=10*np.log10((y[:m*frame]**2).reshape(m,frame).mean(1)+1e-20)
    floor=np.percentile(e,20)
    op=e>(floor+margin_db)
    op=np.convolve(op.astype(float), np.ones(pre+post+1), mode='same')>0   # hangover
    ramp=np.convolve(op.astype(float), np.ones(5)/5, mode='same')
    g=10**(gate_db/20)+(1-10**(gate_db/20))*np.clip(ramp,0,1)
    return y[:m*frame]*np.repeat(g,frame)

d16=lambda x: resample_poly(x,1,3)
def act(x,thr=-35):
    n=int(0.02*FS16); m=len(x)//n
    e=20*np.log10(np.sqrt((x[:m*n]**2).reshape(m,n).mean(1))+1e-12); return np.repeat(e>e.max()+thr,n)
def norm(y,a,t=-18.):
    k=min(len(y),len(a)); return y*(10**(t/20)/(np.sqrt(np.mean(y[:k][a[:k]]**2)+1e-20)))
def fr(y,n=320):
    m=len(y)//n; return (y[:m*n]**2).reshape(m,n).mean(1)

sp=load_speech(6); mics=GEOM['boom_50mm']; d=.05; ctr=.5*(mics[0]+mics[1])
u=lambda z: z/(np.sqrt(np.mean(z**2))+1e-12)
near=np.concatenate([u(sp[0]),u(sp[1])]); far=np.concatenate([u(sp[3]),u(sp[4])])[:len(near)]
na,_=sf.read('partG_real_eval/ex02_noisy.wav'); ca,_=sf.read('partG_real_eval/ex02_clean.wav')
bg=np.resize(resample_poly((na-ca).astype(np.float64),3,1),len(near))
src_n=ctr+(0.05+d/2)*np.array([-1.,0.]); th=np.deg2rad(120.)
src_f=ctr+2.0*np.array([np.cos(th),np.sin(th)])
t1,t2,_=render(near,src_n,mics,rt60=.45,r_crit=1.5,seed=11)
i1,i2,_=render(far,src_f,mics,rt60=.45,r_crit=1.5,seed=12)
b1,b2=diffuse_pair(bg,d,rt60=.6,seed=13)
a_=np.sqrt(np.mean(t1**2)/np.mean(i1**2)); i1*=a_; i2*=a_
a_=np.sqrt(np.mean(t1**2)/np.mean(b1**2)/10**.5); b1*=a_; b2*=a_
m1,m2=d16(t1+i1+b1),d16(t2+i2+b2); ref=d16(t1); n=len(m1); A=act(ref)
_,_,M1=stft2(m1,nfft=512,hop=128); _,_,M2=stft2(m2,nfft=512,hop=128)
il=20*np.log10((np.abs(M1)+1e-9)/(np.abs(M2)+1e-9))
g=0.05+0.95/(1.+np.exp(-(il-3.0)/1.0))
mk=istft2(g*M1,nfft=512,hop=128,n=n)
outs={'1_BEFORE':norm(m1,A),
      '2_AFTER_mask_L1':norm(L1(norm(mk,A)),A),
      '3_AFTER_plus_energy_gate':norm(squelch(L1(norm(mk,A))),A),
      '4_AFTER_plus_NEARFIELD_gate':norm(nf_gate(L1(norm(mk,A)),il,M1),A),
      '5_REFERENCE':norm(ref,A)}
er=fr(ref); quiet=er<=np.quantile(er,0.05)   # frames where the WEARER is truly silent
print(f"\n8.0 s scene  {'stage':24}{'PESQ':>7}{'STOI':>7}{'SI-SDR':>8}{'floor between words':>21}")
for k,y in outs.items():
    kk=min(len(y),len(ref)); r_,y_=ref[:kk]/np.abs(ref).max(),y[:kk]/(np.abs(y).max()+1e-9)
    e=fr(y); q=min(len(e),len(quiet)); nf=10*np.log10(e[:q][quiet[:q]].mean()+1e-20)
    print(f"{'':13}{k:24}{pesq_fn(FS16,r_,y_,'wb'):>7.3f}{stoi_fn(r_,y_,FS16,extended=False):>7.3f}"
          f"{10*np.log10(np.mean(r_**2)/(np.mean((y_-r_*np.dot(y_,r_)/np.dot(r_,r_))**2)+1e-20)):>8.2f}{nf:>16.1f} dB")
for k,y in outs.items():
    w=f'{OUTD}/{k}.wav'; sf.write(w,np.clip(y,-1,1).astype(np.float32),FS16)
    os.system(f'ffmpeg -y -loglevel error -i "{w}" -codec:a libmp3lame -q:a 2 "{OUTD}/{k}.mp3"')
print("\nwritten",OUTD,"| %.1f s"%(n/FS16))
