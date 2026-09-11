#!/usr/bin/env python3
"""Run the real GTCRNLite for the ESP32's 'N' mode.

    python3 neural_ab.py /dev/cu.usbmodem1401

The ESP32 records 5 s, sends it up, this runs the network, sends it back, and
the ESP32 plays RAW then NEURAL through the speaker. The network runs here,
not on the chip -- see cmdNeural() in the sketch for why.
"""
import sys, time, warnings, numpy as np, serial, torch
warnings.filterwarnings("ignore")
E03 = "/Users/mariyafatima/PS#2/E03"
sys.path[:0] = [E03, E03 + "/model"]
from eval_stratified import load_model
from train_interim import stft as tstft, istft as tistft, DEV, apply_deep_filter
from gtcrn_lite import N_FFT

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem1401"
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6
model, _ = load_model(f"{E03}/runs/g12_noise232/best.pt"); model.eval()
win = torch.hann_window(N_FFT, device=DEV)
hop = int(getattr(model, "hop_", torch.tensor(256)).item())

def enhance(x):
    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32)).to(DEV)[None]
        X = tstft(t, win, hop)
        mm, mp, spp, dfc = model(torch.stack([X.real, X.imag], 1).transpose(2, 3))
        mm = torch.clamp(mm.clamp(min=0.) ** ALPHA, min=0.01)
        mag = X.abs().transpose(1, 2).unsqueeze(1)
        ph = X.angle().transpose(1, 2).unsqueeze(1)
        if mp is not None: ph = ph + mp
        Se = ((mm * mag) * torch.exp(1j * ph)).squeeze(1).transpose(1, 2)
        if dfc is not None: Se = apply_deep_filter(Se, dfc, X, model.df_taps, model.df_bins)
        return tistft(Se, win, t.shape[-1], hop)[0].cpu().numpy()[:len(x)]

ser = serial.Serial(PORT, 2000000, timeout=60)
time.sleep(2.0); ser.reset_input_buffer()
ser.write(b"N")
print(f"sent N (alpha {ALPHA}) -- talk for 5 seconds")

n = None
while True:
    line = ser.readline().decode("utf-8", "replace").strip()
    if not line: continue
    print("  ", line)
    if line.startswith("NEURAL"):
        n = int(line.split()[1]); break

need, buf = n * 2, b""
while len(buf) < need:
    c = ser.read(need - len(buf))
    if not c: raise RuntimeError(f"short read {len(buf)}/{need}")
    buf += c
x = np.frombuffer(buf, "<i2").astype(np.float64) / 32768.0
print(f"  got {n} samples, running GTCRNLite ...")

y = enhance(x)
# match speech level to the input so only the background differs
fr = 320; m = len(x) // fr
ex = (x[:m*fr]**2).reshape(m, fr).mean(1); ey = (y[:m*fr]**2).reshape(m, fr).mean(1)
loud = ex >= np.quantile(ex, 0.70)
y *= np.sqrt(ex[loud].mean() / (ey[loud].mean() + 1e-20))
y = np.clip(y, -0.99, 0.99)
ser.write((y * 32000).astype("<i2").tobytes()); ser.flush()
print("  sent back, listen for RAW then NEURAL")
while True:
    line = ser.readline().decode("utf-8", "replace").strip()
    if line: print("  ", line)
    if line.startswith("done"): break
ser.close()
