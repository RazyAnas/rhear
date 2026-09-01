# RHEAR L1 — training handoff

One job: train the L1 speech enhancer **without the phase branch** and send
back three files. Everything needed is in this folder. No dataset building, no
downloads.

Expect **1–2 hours on a modern CUDA card**; ~6 hours on an Apple M-series.

Why a GPU helps here is worth knowing, because it is not about raw FLOPS.
Measured on this model, **76% of each training step is the DualPathRNN**, whose
time-axis GRU runs 1001 sequential frames per clip — frame 500 cannot start
before 499 finishes. Effective throughput is only ~16 GMAC/s. PyTorch on CUDA
uses **cuDNN's fused RNN kernels**, which collapse exactly those sequential
steps; MPS has no equivalent. That fusion is the speedup, not the card's
peak throughput.

The 3 GB of data is not the cost — it is read into RAM once at startup.

---

## 1. Setup

Needs Python 3.10+ and a CUDA GPU with **≥ 6 GB RAM free on the host** (the
training set is cached in system RAM as int16, ~5.1 GB — not GPU memory).

```bash
pip install torch numpy soundfile pystoi pesq
```

The device is picked automatically: **CUDA → MPS → CPU**. Check it says `cuda`
before letting it run — on CPU this takes days, and it will not warn you.

```bash
cd code
python3 -c "import sys;sys.path.insert(0,'.');from train_interim import DEV;print(DEV)"
```

If that prints `cpu` on a GPU machine, the PyTorch build has no CUDA. Fix that
first. `RHEAR_DEVICE=cuda` forces it.

---

## 2. The one command

```bash
cd code
python3 -u train_interim.py \
    --data ../data/h3_20k \
    --epochs 24 \
    --batch 32 \
    --cache-int16 \
    --no-phase \
    --out runs/nophase \
    2>&1 | tee ../train_nophase.log
```

**Do not change any of these values.** They are the frozen protocol from the
run this is compared against — same data, same splits, same optimizer, same
budget. Changing one makes the comparison meaningless.

`--cache-int16` loads the whole training set into RAM as int16. If the machine
has plenty of RAM you may instead use `--workers 8` without it, which streams
from disk; both are equivalent numerically, only speed differs.

---

## 3. Is it working?

Startup should print:

```
device cuda   train 20000   val 300   (cache: int16)
model 22,956 params, 58.79 MMAC/s
```

**22,956 params is the check that `--no-phase` took effect.** With the branch
it is 22,988. If you see 22,988, the flag did not apply — stop and say so.

Progress prints every 5 epochs. For reference, the run this replaces (which
*had* the phase branch) went:

| epoch | val SI-SDR |
|---|---|
| 0 | 7.17 dB |
| 5 | 8.68 |
| 10 | 9.10 |
| 15 | 9.41 |
| 20 | 9.45 |
| 23 | 9.45 (best) |

The no-phase run does not have to match these — that is the experiment. But if
epoch 0 is far below ~7 dB, or `skip` is not `0/625`, something is wrong.

---

## 4. Send back

1. `code/runs/nophase/best.pt` — the checkpoint (~210 KB)
2. `code/runs/nophase/history.json` — per-epoch metrics
3. `train_nophase.log` — the full log

That is all. The evaluation runs on this side.

---

## 5. What is in `data/`

| Folder | What it is | Used for |
|---|---|---|
| `h3_20k/` | 20,000 training + 300 validation mixtures, FLAC | **training** |
| `edef/` | 300-clip defence-noise evaluation set, 8 classes | evaluation only |
| `realnoise/` | 300-clip environmental evaluation set | evaluation only |

The two eval sets are shipped so results can be reproduced, but the training
command above never touches them. Every noise source in them is verified
disjoint from training — do not mix them in.

Audio is FLAC, PCM_16, 16 kHz, 4 s clips. FLAC is lossless; it is used only
because it halves the size.

---

## 6. If something breaks

| Symptom | Cause |
|---|---|
| device prints `cpu` on a GPU box | PyTorch built without CUDA |
| `LibsndfileError` | a manifest path does not match a file on disk — report it, do not "fix" the manifest |
| params print 22,988 | `--no-phase` did not apply |
| out-of-memory on the host | drop `--cache-int16`, add `--workers 8` |
| `skip` count above 0 | report it; it should be `0/625` every epoch |

Please do not tune the learning rate, batch size, epochs or loss to make the
numbers look better. A worse number from the frozen protocol is a useful
result; a better number from a changed protocol is not.
