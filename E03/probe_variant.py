"""Which variant produced the 24-epoch history?

history.json records only train_loss, so the variant is not stated in it.
But --perceptual ADDS a non-negative term to the same base objective, so the
two variants sit at measurably different loss levels on identical batches.

Train the plain --no-phase model and, at every step, ALSO evaluate the
perceptual term without letting it touch the gradient. That yields, from one
run, both what G0 would record and what G1 would record.
"""
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser("~/PS#2/E03"))
os.chdir(os.path.expanduser("~/PS#2/E03"))
from train_interim import (Pairs, GTCRNLite, loss_fn, perceptual_loss, enhance,
                           stft, count_params, DEV, N_FFT)
from rhear_data.manifest import read_manifest

DATA = os.path.expanduser("~/PS#2/handoff/data/h3_20k")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
torch.manual_seed(0)

hdr, rows = read_manifest(os.path.join(DATA, "manifest.jsonl"))
tr = torch.utils.data.DataLoader(Pairs(DATA, "train", rows, cache=False),
                                 batch_size=32, shuffle=True, num_workers=0,
                                 drop_last=True)
model = GTCRNLite(phase=False, fullband=False).to(DEV)
erb_M = model.erb.M
print(f"  device {DEV}   G0 model {count_params(model):,} params "
      f"(G1 is identical; only the loss differs)")
opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
win = torch.hann_window(N_FFT, device=DEV)

base_s = perc_s = 0.0; n = 0; t0 = time.time()
model.train()
for x, y in tr:
    if n >= STEPS: break
    x, y = x.to(DEV), y.to(DEV)
    if not torch.isfinite(y).all() or float(y.pow(2).mean()) < 1e-9: continue
    Se, est, X, spp = enhance(model, x, win)
    loss, parts = loss_fn(est, y, Se, stft(y, win), spp, perc=0.0, erb_M=None)
    if not torch.isfinite(loss): continue
    with torch.no_grad():
        lp = float(perceptual_loss(est, y, erb_M))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    base_s += float(loss); perc_s += lp; n += 1
    if n % 25 == 0:
        print(f"  step {n:4d}  G0 mean {base_s/n:8.3f}   "
              f"perc mean {perc_s/n:6.3f}   G1 would be {(base_s+perc_s)/n:8.3f}")

b, p = base_s/n, perc_s/n
print(f"\n  over {n} steps ({time.time()-t0:.0f}s)")
print(f"    G0 (--no-phase)                  train_loss ~ {b:8.3f}")
print(f"    G1 (--no-phase --perceptual 1.0) train_loss ~ {b+p:8.3f}   (+{p:.3f})")
print(f"\n    observed epoch-0 train_loss      = {-2.095646674501896:8.3f}")
print(f"      distance to G0: {abs(b-(-2.0956)):7.3f}   to G1: {abs(b+p-(-2.0956)):7.3f}")
