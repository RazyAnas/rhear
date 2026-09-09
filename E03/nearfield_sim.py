#!/usr/bin/env python3
"""
RHEAR - two-microphone near-field / far-field simulation.

Under test (the proposal):
  exploit wavefront CURVATURE at a 2-mic array to recover
     (a) angle, (b) range, (c) a calibrated confidence
  and ask whether it buys anything for the L1 communication layer.

Physics modelled (deliberately, because the first two-mic attempt in this
project was wrong for exactly the opposite reason - it used free-field 1/r
only and produced a fantasy 40 dB input SIR):

  * exact spherical propagation: per-mic delay r/c AND 1/r amplitude.
    The 1/r term IS the near-field cue. A plane-wave model has none.
  * fractional delay via FFT phase ramp (exact for band-limited signals)
  * diffuse late reverberation carrying the theoretical two-mic coherence
    Gamma(f) = sinc(2 pi f d / c), built with Habets' mixing construction
  * direct-to-reverberant ratio falling as 1/r about a critical distance
  * INMP441 self-noise (61 dB SNR re 94 dB SPL) and +-1 dB sensitivity spread
  * both mics on ONE I2S bus (shared BCLK/WS, L/R strapped opposite)
    => sample-synchronous, no clock skew. This is a real and favourable fact.
"""
import json, os, sys, math
import numpy as np
from scipy.signal import resample_poly, stft, istft, fftconvolve
import soundfile as sf

C   = 343.0
FS  = 48000
RNG = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "nearfield")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- propagation
def frac_delay(x, n_delay):
    """Delay by a (fractional) number of samples using an FFT phase ramp."""
    N = len(x)
    X = np.fft.rfft(x)
    k = np.arange(len(X))
    return np.fft.irfft(X * np.exp(-2j*np.pi*k*n_delay/N), n=N)

def direct_path(x, src, mic, fs=FS, r_min=0.02):
    """Exact spherical wave: delay r/c, amplitude 1/r."""
    r = max(float(np.linalg.norm(np.asarray(src) - np.asarray(mic))), r_min)
    return frac_delay(x, r/C*fs) / r, r

def diffuse_pair(x, d, fs=FS, rt60=0.45, seed=0):
    """
    Two-channel diffuse late field with coherence sinc(2 pi f d / c).
    Channel spectra are identical; only their coherence is constrained.
    """
    rg = np.random.default_rng(seed)
    N  = len(x)
    tau = rt60 / 6.91
    t   = np.arange(int(min(rt60*3, 1.5)*fs)) / fs
    def tail():
        h = rg.standard_normal(len(t)) * np.exp(-t/tau)
        h[:int(0.002*fs)] = 0.0                      # late field only
        return fftconvolve(x, h)[:N]
    a, b = tail(), tail()
    A, B = np.fft.rfft(a), np.fft.rfft(b)
    f = np.fft.rfftfreq(N, 1/fs)
    g = np.sinc(2*f*d/C)                             # np.sinc(z)=sin(pi z)/(pi z)
    y1 = a
    y2 = np.fft.irfft(g*A + np.sqrt(np.maximum(1-g**2, 0))*B, n=N)
    return y1, y2

def render(x, src, mics, rt60=0.45, r_crit=1.5, seed=0):
    """One source -> two mics: direct spherical path + range-dependent diffuse tail."""
    d = float(np.linalg.norm(np.asarray(mics[1]) - np.asarray(mics[0])))
    d1, r1 = direct_path(x, src, mics[0])
    d2, r2 = direct_path(x, src, mics[1])
    if r_crit is None or not np.isfinite(r_crit):        # open field, no reverb
        return d1, d2, (r1, r2)
    drr_db = 20*np.log10(max(r_crit, 1e-6)/max(0.5*(r1+r2), 1e-6))
    e_dir  = np.mean(d1**2) + 1e-20
    v1, v2 = diffuse_pair(x, d, rt60=rt60, seed=seed)
    e_rev  = np.mean(v1**2) + 1e-20
    k = np.sqrt(e_dir / e_rev * 10**(-drr_db/10))
    return d1 + k*v1, d2 + k*v2, (r1, r2)

def sensor_noise(n, ref_rms, snr_db=61.0, seed=0):
    rg = np.random.default_rng(seed)
    return rg.standard_normal(n) * ref_rms * 10**(-snr_db/20)

# ------------------------------------------------------------------ estimators
def gcc_phat(x1, x2, fs=FS, max_tau=None, interp=16):
    n = len(x1) + len(x2)
    X1, X2 = np.fft.rfft(x1, n), np.fft.rfft(x2, n)
    R = X1 * np.conj(X2)
    R /= (np.abs(R) + 1e-12)
    cc = np.fft.irfft(R, n=n*interp)
    mx = int(interp * n/2)
    if max_tau is not None:
        mx = min(mx, int(interp*fs*max_tau))
    cc = np.concatenate((cc[-mx:], cc[:mx+1]))
    pk = int(np.argmax(np.abs(cc)))
    tau = (pk - mx) / float(interp*fs)
    # peak-to-sidelobe ratio, excluding a guard band around the peak
    guard = max(1, int(interp*fs*2e-5))
    m = np.ones(len(cc), bool); m[max(0,pk-guard):pk+guard+1] = False
    psr = float(np.abs(cc[pk]) / (np.abs(cc[m]).max() + 1e-12))
    return tau, psr

def ild_db(x1, x2, active):
    a = np.sqrt(np.mean(x1[active]**2) + 1e-20)
    b = np.sqrt(np.mean(x2[active]**2) + 1e-20)
    return 20*np.log10(a/b)

def joint_estimate(tau_meas, ild_meas, mics, r_grid, th_grid,
                   sig_tau=8e-6, sig_ild=0.5):
    """
    Solve for (range, angle) from BOTH cues at once - the proposal, done exactly.
    Returns the MAP point and the full negative-log cost surface.
    """
    m0, m1 = np.asarray(mics[0]), np.asarray(mics[1])
    ctr = 0.5*(m0+m1)
    R, TH = np.meshgrid(r_grid, th_grid, indexing="ij")
    px = ctr[0] + R*np.cos(TH); py = ctr[1] + R*np.sin(TH)
    r0 = np.sqrt((px-m0[0])**2 + (py-m0[1])**2)
    r1 = np.sqrt((px-m1[0])**2 + (py-m1[1])**2)
    tau_pred = (r0 - r1)/C
    ild_pred = 20*np.log10(r1/np.maximum(r0, 1e-9))
    cost = ((tau_pred-tau_meas)/sig_tau)**2 + ((ild_pred-ild_meas)/sig_ild)**2
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return float(r_grid[i]), float(th_grid[j]), cost

def profile_width(cost, grid, axis, drop=1.0):
    """Width of the {cost <= min+drop} set along one axis -> a real uncertainty."""
    prof = cost.min(axis=1 if axis == 0 else 0)
    ok = prof <= prof.min() + drop
    if not ok.any(): return float("nan")
    idx = np.where(ok)[0]
    return float(grid[idx[-1]] - grid[idx[0]])

# --------------------------------------------------------------------- corpus
def load_speech(n_want=6):
    import glob
    fs_files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "partG_real_eval", "*_clean.wav")))
    out = []
    for f in fs_files[:n_want]:
        x, sr = sf.read(f)
        x = x.astype(np.float64)
        if x.ndim > 1: x = x.mean(1)
        x = resample_poly(x, FS//math.gcd(FS, sr), sr//math.gcd(FS, sr))
        x /= (np.abs(x).max() + 1e-12)
        out.append(x)
    return out

def vad(x, fs=FS, frame=0.02, thr_db=-35):
    n = int(frame*fs); m = len(x)//n
    e = 20*np.log10(np.sqrt((x[:m*n]**2).reshape(m, n).mean(1)) + 1e-12)
    keep = e > (e.max() + thr_db)
    return np.repeat(keep, n)

# -------------------------------------------------------------------- geometry
def boom(d):
    """Boom pair: mic0 nearest the mouth, mic1 a distance d further along +x."""
    return [np.array([0.0, 0.0]), np.array([d, 0.0])]

GEOM = {"boom_15mm": boom(0.015), "boom_50mm": boom(0.050), "ears_150mm": boom(0.150)}

# =============================================================== experiments
def diffuse_bg(n, d, seed):
    rg = np.random.default_rng(seed)
    w = rg.standard_normal(n)
    return diffuse_pair(w, d, rt60=0.6, seed=seed+1)

def mic_pair(src_xy, speech, geom, snr_db, rt60, r_crit, seed, tol_db=1.0, calibrate=False):
    mics = GEOM[geom]; d = float(np.linalg.norm(mics[1]-mics[0]))
    y1, y2, (r0, r1) = render(speech, src_xy, mics, rt60=rt60, r_crit=r_crit, seed=seed)
    n1, n2 = diffuse_bg(len(y1), d, seed=seed+100)
    e_s = np.mean(y1**2) + 1e-20
    k = np.sqrt(e_s / (np.mean(n1**2) + 1e-20) * 10**(-snr_db/10))
    y1 = y1 + k*n1; y2 = y2 + k*n2
    rg = np.random.default_rng(seed + 999)
    g = 10**(rg.uniform(-tol_db, tol_db, 2)/20)
    if calibrate: g[:] = 1.0
    y1 *= g[0]; y2 *= g[1]
    ref = np.sqrt(np.mean(y1**2))
    y1 += sensor_noise(len(y1), ref, seed=seed+5)
    y2 += sensor_noise(len(y2), ref, seed=seed+6)
    return y1, y2, (r0, r1)

R_GRID  = np.concatenate([np.arange(0.02, 0.50, 0.005), np.arange(0.50, 12.0, 0.05)])
TH_GRID = np.deg2rad(np.arange(0.0, 180.5, 0.5))

def exp_angle_range(geoms=("boom_15mm", "boom_50mm", "ears_150mm"),
                    ranges=(0.03, 0.06, 0.15, 0.5, 1.0, 2.0, 4.0, 6.0),
                    angles=(150.0, 120.0, 90.0, 60.0, 30.0),
                    snrs=(20.0, 10.0, 0.0), rt60=0.45, r_crit=1.5, trials=None):
    speech = load_speech()
    rows = []
    for geom in geoms:
        mics = GEOM[geom]; d = float(np.linalg.norm(mics[1]-mics[0]))
        ctr = 0.5*(mics[0]+mics[1])
        for r in ranges:
            for th_deg in angles:
                th = np.deg2rad(th_deg)
                src = ctr + r*np.array([np.cos(th), np.sin(th)])
                for snr in snrs:
                    for si, s in enumerate(speech):
                        y1, y2, (r0, r1) = mic_pair(src, s, geom, snr, rt60, r_crit,
                                                    seed=1000*si + int(r*100) + int(th_deg))
                        a = vad(s)[:len(y1)]
                        tau, psr = gcc_phat(y1[a], y2[a], max_tau=(d/C)*1.6)
                        il = ild_db(y1, y2, a)
                        rh, thh, cost = joint_estimate(tau, il, mics, R_GRID, TH_GRID)
                        rows.append(dict(
                            geom=geom, d_mm=d*1000, r_true=r, th_true=th_deg, snr=snr, clip=si,
                            tau_us=tau*1e6, tau_true_us=(r0-r1)/C*1e6, psr=psr,
                            ild=il, ild_true=20*np.log10(r1/r0),
                            r_hat=rh, th_hat=np.rad2deg(thh),
                            r_width=profile_width(cost, R_GRID, 0),
                            th_width=np.rad2deg(profile_width(cost, TH_GRID, 1))))
    return rows

# ------------------------------------------------- near/far discrimination
def exp_nearfar(geoms=("boom_15mm","boom_50mm","ears_150mm"),
                near_r0=(0.03,0.05,0.08), far_r=(0.5,1.0,2.0,4.0),
                angles=(180.0,150.0,120.0), snr=10.0, rt60=0.45, r_crit=1.5,
                calibrate=False):
    speech = load_speech(); rows=[]
    for geom in geoms:
        mics = GEOM[geom]; d=float(np.linalg.norm(mics[1]-mics[0])); ctr=0.5*(mics[0]+mics[1])
        cases = [("wearer", r0+d/2, 180.0) for r0 in near_r0] + \
                [("bystander", r, a) for r in far_r for a in angles]
        for lab, r, th_deg in cases:
            th=np.deg2rad(th_deg); src = ctr + r*np.array([np.cos(th), np.sin(th)])
            for si,s in enumerate(speech):
                y1,y2,(r0,r1) = mic_pair(src, s, geom, snr, rt60, r_crit,
                                         seed=2000*si+int(r*100)+int(th_deg),
                                         calibrate=calibrate)
                a = vad(s)[:len(y1)]
                rows.append(dict(geom=geom, cls=lab, r=r, th=th_deg, clip=si,
                                 ild=ild_db(y1,y2,a), r0=r0, r1=r1))
    return rows

# ------------------------------------------------------------- payoff (mask)
def stft2(x, nfft=1024, hop=256):
    f,t,Z = stft(x, fs=FS, window="hann", nperseg=nfft, noverlap=nfft-hop,
                 boundary=None, padded=False)
    return f,t,Z

def istft2(Z, nfft=1024, hop=256, n=None):
    _,x = istft(Z, fs=FS, window="hann", nperseg=nfft, noverlap=nfft-hop,
                boundary=False)
    return x[:n] if n else x

def exp_payoff(geom="boom_50mm", wearer_r0=0.05, byst_r=2.0, byst_th=120.0,
               in_sirs=(-5.,0.,5.,10.), rt60=0.45, r_crit=1.5,
               thr_db=None, slope=2.0, calibrate=False):
    speech = load_speech()
    mics = GEOM[geom]; d=float(np.linalg.norm(mics[1]-mics[0])); ctr=0.5*(mics[0]+mics[1])
    src_t = ctr + (wearer_r0+d/2)*np.array([np.cos(np.pi), np.sin(np.pi)])
    th=np.deg2rad(byst_th); src_i = ctr + byst_r*np.array([np.cos(th), np.sin(th)])
    if thr_db is None:
        rt0=np.linalg.norm(src_t-mics[0]); rt1=np.linalg.norm(src_t-mics[1])
        thr_db = 0.5*20*np.log10(rt1/rt0)          # halfway between wearer and 0 dB
    rows=[]
    for si in range(len(speech)):
        s_t = speech[si]; s_i = speech[(si+3) % len(speech)][:len(s_t)]
        s_i = np.pad(s_i,(0,max(0,len(s_t)-len(s_i))))[:len(s_t)]
        t1,t2,(rt0,rt1) = render(s_t, src_t, mics, rt60=rt60, r_crit=r_crit, seed=si)
        i1,i2,(ri0,ri1) = render(s_i, src_i, mics, rt60=rt60, r_crit=r_crit, seed=si+50)
        geo_sir = 10*np.log10(np.mean(t1**2)/(np.mean(i1**2)+1e-20))
        rg=np.random.default_rng(si+3); g=10**(rg.uniform(-1,1,2)/20)
        if calibrate: g[:]=1.0
        for sir in in_sirs:
            a = np.sqrt(np.mean(t1**2)/(np.mean(i1**2)+1e-20)/10**(sir/10))
            I1,I2 = a*i1, a*i2
            m1 = (t1+I1)*g[0]; m2 = (t2+I2)*g[1]
            f,_,M1 = stft2(m1); _,_,M2 = stft2(m2)
            ild = 20*np.log10((np.abs(M1)+1e-9)/(np.abs(M2)+1e-9))
            mask = 1.0/(1.0+np.exp(-(ild-thr_db)/slope))
            n=len(m1)
            _,_,T1 = stft2(t1*g[0]); _,_,J1 = stft2(I1*g[0])
            t_out = istft2(mask*T1, n=n); i_out = istft2(mask*J1, n=n)
            sir_out = 10*np.log10((np.mean(t_out**2)+1e-20)/(np.mean(i_out**2)+1e-20))
            sir_in  = 10*np.log10((np.mean((t1*g[0])**2)+1e-20)/(np.mean((I1*g[0])**2)+1e-20))
            keep = 10*np.log10((np.mean(t_out**2)+1e-20)/(np.mean((t1*g[0])**2)+1e-20))
            rows.append(dict(clip=si, in_sir=sir, sir_in=sir_in, sir_out=sir_out,
                             d_sir=sir_out-sir_in, target_loss_db=keep,
                             geo_sir=geo_sir, thr_db=thr_db))
    return rows

# ------------------------------------------------------------------- report
def agg(rows, keys, val, fn=np.median):
    out={}
    for r in rows:
        k=tuple(r[x] for x in keys); out.setdefault(k,[]).append(r[val])
    return {k: fn(v) for k,v in out.items()}

def main():
    res={}
    print("="*78); print("RHEAR two-mic near-field simulation"); print("="*78)

    print("\n[1/4] angle + range estimation ...", flush=True)
    ar = exp_angle_range(); res["angle_range"]=ar

    print("\n--- ANGLE error (median |deg|), sources at r >= 0.5 m ---")
    print(f"{'geometry':12}{'aperture':>10}" + "".join(f"{f'SNR {s:g} dB':>12}" for s in (20,10,0)))
    for g in ("boom_15mm","boom_50mm","ears_150mm"):
        line=f"{g:12}{[r for r in ar if r['geom']==g][0]['d_mm']:>8.0f}mm"
        for s in (20,10,0):
            e=[abs(r['th_hat']-r['th_true']) for r in ar
               if r['geom']==g and r['snr']==s and r['r_true']>=0.5]
            line+=f"{np.median(e):>12.1f}"
        print(line)

    print("\n--- RANGE: is it identifiable at all? (SNR 20 dB, best case) ---")
    print(f"{'geometry':12}{'r_true':>8}{'r_hat':>9}{'rel.err':>9}{'68% width of r':>17}")
    for g in ("boom_50mm","ears_150mm"):
        for rt in (0.03,0.06,0.15,0.5,1.0,2.0,4.0,6.0):
            sub=[r for r in ar if r['geom']==g and r['snr']==20 and abs(r['r_true']-rt)<1e-9]
            if not sub: continue
            rh=np.median([r['r_hat'] for r in sub]); w=np.median([r['r_width'] for r in sub])
            print(f"{g:12}{rt:>8.2f}{rh:>9.2f}{(rh-rt)/rt*100:>8.0f}%"
                  f"{('  +-%.2f m'%(w/2)) if np.isfinite(w) else '   unbounded':>17}")

    print("\n[2/4] wearer vs bystander discrimination ...", flush=True)
    nf = exp_nearfar(); res["nearfar"]=nf
    print("\n--- level ratio between the two mics (dB), UNCALIBRATED (+-1 dB parts) ---")
    print(f"{'geometry':12}{'wearer ILD':>22}{'bystander ILD':>22}{'separation':>12}{'acc':>7}")
    for g in ("boom_15mm","boom_50mm","ears_150mm"):
        w=np.array([r['ild'] for r in nf if r['geom']==g and r['cls']=='wearer'])
        b=np.array([r['ild'] for r in nf if r['geom']==g and r['cls']=='bystander'])
        dp=(w.mean()-b.mean())/np.sqrt(0.5*(w.var()+b.var())+1e-12)
        ths=np.linspace(min(b.min(),w.min()), max(b.max(),w.max()), 400)
        acc=max(((w>t).mean()+(b<=t).mean())/2 for t in ths)
        print(f"{g:12}{w.mean():>15.2f} +-{w.std():<5.2f}{b.mean():>14.2f} +-{b.std():<5.2f}"
              f"{dp:>10.1f} d'{acc*100:>6.1f}%")

    print("\n[3/4] payoff: near-field mask on wearer + bystander ...", flush=True)
    pay={}
    for g in ("boom_15mm","boom_50mm"):
        pay[g]=exp_payoff(geom=g)
    res["payoff"]=pay
    print("\n--- SIR gain from the near-field mask (boom pair, bystander at 2 m, RT60 0.45 s) ---")
    print(f"{'geometry':12}{'in SIR':>9}{'out SIR':>10}{'gain':>9}{'target loss':>13}")
    for g,rows in pay.items():
        for s in (-5.,0.,5.,10.):
            sub=[r for r in rows if r['in_sir']==s]
            print(f"{g:12}{np.median([r['sir_in'] for r in sub]):>9.1f}"
                  f"{np.median([r['sir_out'] for r in sub]):>10.1f}"
                  f"{np.median([r['d_sir'] for r in sub]):>+9.1f}"
                  f"{np.median([r['target_loss_db'] for r in sub]):>+12.1f} dB")
    print(f"\n  free geometric SIR (wearer 5 cm vs bystander 2 m, equal source level): "
          f"{np.median([r['geo_sir'] for r in pay['boom_50mm']]):+.1f} dB")

    print("\n[4/4] open field (no reverb) control ...", flush=True)
    of=exp_payoff(geom="boom_50mm", r_crit=None); res["payoff_openfield"]=of
    for s in (0.,10.):
        sub=[r for r in of if r['in_sir']==s]
        print(f"  open field  in {np.median([r['sir_in'] for r in sub]):>5.1f} dB"
              f" -> gain {np.median([r['d_sir'] for r in sub]):>+5.1f} dB")

    with open(os.path.join(OUT,"nearfield_results.json"),"w") as f:
        json.dump({k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items()})
                   for k,v in res.items()}, f, default=float)
    print(f"\nwritten {os.path.join(OUT,'nearfield_results.json')}")

if __name__ == "__main__":
    main()
