#!/usr/bin/env python3
"""Export G13b to a flat float32 blob + a C header of offsets.

Stage 1 of putting the model on the ESP32. Everything downstream reads this
one file, so the C never has to know PyTorch's naming or ordering.

    /opt/anaconda3/bin/python port_c/export_weights.py
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
E03 = os.path.dirname(HERE)
sys.path[:0] = [E03, E03 + "/model"]
from eval_stratified import load_model

CKPT = f"{E03}/runs/g13b_comms/best.pt"
model, (has_phase, has_fb) = load_model(CKPT)
sd = model.state_dict()

blob, offs, off = [], {}, 0
for k in sorted(sd.keys()):
    v = sd[k].detach().cpu().numpy().astype(np.float32).ravel()
    if v.size == 0:
        continue
    offs[k] = {"off": off, "n": int(v.size), "shape": list(sd[k].shape)}
    blob.append(v); off += v.size

flat = np.concatenate(blob)
with open(f"{HERE}/g13b_weights.bin", "wb") as f:
    flat.tofile(f)
with open(f"{HERE}/g13b_weights.json", "w") as f:
    json.dump({"ckpt": CKPT, "total": int(flat.size), "phase": has_phase,
               "fullband": has_fb, "hop": int(model.hop_.item()),
               "tensors": offs}, f, indent=1)

print(f"{flat.size:,} float32 = {flat.nbytes/1024:.1f} KB fp32"
      f"  ({flat.nbytes/4/1024:.1f} KB at int8)")
print(f"wrote g13b_weights.bin + .json  ({len(offs)} tensors)")

# reference activations for numeric validation of the C, on real audio
import soundfile as sf
from train_interim import stft as tstft, DEV
from gtcrn_lite import N_FFT
x, _ = sf.read("/tmp/real_in.wav")
x = x[:16000 * 2].astype(np.float32)
win = torch.hann_window(N_FFT, device=DEV)
with torch.no_grad():
    t = torch.from_numpy(x).to(DEV)[None]
    X = tstft(t, win, int(model.hop_.item()))
    spec = torch.stack([X.real, X.imag], 1).transpose(2, 3)
    mm, mp, spp, dfc = model(spec)
np.save(f"{HERE}/ref_spec.npy", spec.cpu().numpy())
np.save(f"{HERE}/ref_mask.npy", mm.cpu().numpy())
print(f"reference: spec {tuple(spec.shape)} -> mask {tuple(mm.shape)}")
