# RHEAR — two more training runs (G1, G2)

Thanks for running the first one. This is **two more runs on the same data you
already have**. Nothing to download again — the dataset is unchanged.

**Total: ~2–4 hours on a CUDA card, both runs.**

---

## What you already have vs what is new

| | |
|---|---|
| **Already have** | the `handoff/` folder: `data/h3_20k` (2.6 GB), `data/edef`, `data/realnoise` |
| **New here** | `code/` only — updated training script and model |
| **Data** | **identical, do not re-download** |

The code changed since the first handoff: two new options were added, and the
device picker now finds CUDA properly. Use **this** `code/`, not the old one.

---

## Setup — 2 minutes

Put this folder next to your existing `handoff/`, so the layout is:

```
handoff/          <- you already have this
  data/
    h3_20k/       <- 20,000 training mixtures
    edef/
    realnoise/
handoff-g12/      <- this folder
  code/
  README.md
```

Same packages as before:

```bash
pip install torch numpy soundfile
```

**Check CUDA is actually found before starting.** On CPU these take days and
nothing warns you:

```bash
cd handoff-g12/code
python3 -c "import sys;sys.path.insert(0,'.');from train_interim import DEV;print(DEV)"
```

Must print `cuda`. If it prints `cpu` on a GPU machine, your PyTorch has no
CUDA build — fix that first. `RHEAR_DEVICE=cuda` forces it.

---

## Run 1 — G1, perceptual loss

```bash
cd handoff-g12/code
python3 -u train_interim.py \
    --data ../../handoff/data/h3_20k \
    --epochs 24 --batch 32 --cache-int16 \
    --no-phase --perceptual 1.0 \
    --out runs/g1_perceptual 2>&1 | tee ../g1_perceptual.log
```

## Run 2 — G2, full-band branch

```bash
cd handoff-g12/code
python3 -u train_interim.py \
    --data ../../handoff/data/h3_20k \
    --epochs 24 --batch 32 --cache-int16 \
    --no-phase --fullband \
    --out runs/g2_fullband 2>&1 | tee ../g2_fullband.log
```

Run them **one at a time**, not together — two jobs on one GPU make both slower
and make the timings meaningless.

**Do not change any other value.** Data, epochs, batch, optimizer and splits
must match the first run exactly, or the comparison is worthless. Each run
changes exactly one thing.

---

## The check that matters — 30 seconds after each start

The first two lines tell you whether the right thing is training:

| run | must print | if you see this instead |
|---|---|---|
| **G1** | `model 22,956 params` | `22,988` → `--no-phase` did not apply, **stop** |
| **G2** | `model 24,975 params` | `22,956` → `--fullband` did not apply, **stop** |

**The parameter count CANNOT tell G0 and G1 apart.** `--perceptual` changes the
loss, not the model — both are 22,956 params. So the table above does not catch
a dropped `--perceptual`, and a G1 run that silently trained without it looks
completely normal. This has already happened once. Use the loss line instead:

| | `perc` in the loss breakdown | epoch-0 `train_loss` |
|---|---|---|
| `--perceptual` **took** | roughly **10–17** | about **+16** |
| `--perceptual` **did NOT take** | **0.000** | about **−2** |

Those two are an order of magnitude apart, so one glance at epoch 0 settles it.
If `perc` is 0, the flag did not apply — stop and rerun. If it is in the
hundreds, stop and tell me: that is the instability I fixed and I want to know
if it came back.

**Expect G1's val SI-SDR to be LOWER than the run before it.** That is not a
failed run. Perceptual loss deliberately trades SI-SDR for perceptual quality;
PESQ is what it exists to improve, and PESQ is not in `history.json`. Send the
checkpoint and let the evaluation decide — please do not stop a run or retune
anything because val SI-SDR dropped.

`skip` should stay `0/625` every epoch in both runs. Anything else, tell me.

For reference, the first run went 7.17 dB at epoch 0 → 9.45 dB best. These do
not have to match — that is the experiment — but if epoch 0 is far below ~7 dB
something is wrong.

---

## Send back

Six files, about 1 MB total:

```
runs/g1_perceptual/best.pt
runs/g1_perceptual/history.json
g1_perceptual.log

runs/g2_fullband/best.pt
runs/g2_fullband/history.json
g2_fullband.log
```

Evaluation happens on my side. Please also send the same three from the **first**
run if you have not already — I cannot compare these against it otherwise, and
that comparison is the whole point.

---

## What these are testing, briefly

The problem statement asks for two things we did not have. Each run tests one.

**G1 — perceptual loss.** Our training objective had nothing in it that
corresponds to how speech *sounds*. That is the likeliest reason a big gain in
signal-to-noise ratio bought almost no gain in perceived quality. This adds a
loss term that measures error the way hearing does.

**G2 — full-band branch.** Right now the model squeezes 257 frequency slices
into 48 bands before making any decision, so it only ever sees fine detail
through that coarse grid. This adds a second path that keeps all 257 and merges
the two. Costs 7% more compute and 2,019 more parameters.

Either may not help. That is a real outcome and worth knowing — we have already
rejected four ideas this way, and knowing what does not work has been more
useful than guessing.

---

## If something breaks

| symptom | cause |
|---|---|
| device prints `cpu` | PyTorch built without CUDA |
| wrong parameter count | the flag did not apply — check for typos in the command |
| out of memory on the **host** (not GPU) | drop `--cache-int16`, add `--workers 8` |
| `perc` in the hundreds | tell me, do not work around it |
| `skip` above 0 | tell me the number and the epoch |

Please do not tune the learning rate, batch size, epochs or loss to make numbers
look better. A worse number from the frozen setup is useful; a better number
from a changed setup tells us nothing.
