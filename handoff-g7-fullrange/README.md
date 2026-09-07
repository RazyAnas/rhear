# G7 fine-tune, take two — full SNR range

## What happened, in one table

Your `g7_finetune_ms_snsd_0db` checkpoint (hop256 + 50k params + deep-filter,
trained on MS-SNSD + MAD + DEMAND) was evaluated against G7-base on our
standard held-out sets — the metrics your own `evaluate_g7_stoi.py` doesn't
report (PESQ, SI-SIR, SI-SAR):

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def, your checkpoint − G7-base | −0.022 | **−0.109** | −1.83 | +0.45 | **−2.27** |
| realnoise, your checkpoint − G7-base | −0.013 | **−0.050** | −0.46 | +1.05 | **−0.92** |

**SI-SIR is up on both sets — it suppresses more noise. SI-SAR is down hard —
it damages more speech.** Net: worse on PESQ, STOI and SI-SDR everywhere. Under
the project's standing rule (keep only if PESQ improves and STOI/SI-SAR don't
regress, on both sets) this is a clear reject.

**Your architecture instinct was right** — combining hop256+capacity with the
deep-filter cascade and more diverse noise is exactly the direction the
project's oracle-ceiling analysis pointed at. The likely cause is narrower:
your dataset trains at **exactly 0 dB SNR**, a single fixed point, where H3 and
every other checkpoint in this project trains on **−10..+20 dB**. A model that
only ever sees the loudest, most adversarial noise condition learns to always
fight hard — which is exactly the SI-SIR-up / SI-SAR-down signature measured
above.

**This handoff changes exactly that one thing.** Same noise sources, same 70/30
MAD+DEMAND/MS-SNSD mix, same architecture, same seeds — only the SNR range.

## Files

```
code/make_g7_finetune_fullrange.py       your make_g7_finetune_ms_snsd.py,
                                          SNR fixed to -10..20 dB, writes to a
                                          NEW folder (your 0dB dataset is untouched)
code/add_g7_finetune_val_fullrange.py    the matching validation-set builder
```

Drop both into your `E03/` folder next to your originals — they import
`make_g7_realnoise` the same way yours do and need nothing else.

## Run it

```bash
cd E03
python3 make_g7_finetune_fullrange.py
python3 add_g7_finetune_val_fullrange.py
```

**Check before you trust it:** the first script should print
`SNR range: -10.0 .. 20.0 dB` near the top. If it prints `0.0 .. 0.0`, the
override didn't take — stop and check the file didn't get swapped for the
original by accident.

## Train

Two options. **Use A unless you have a specific reason not to.**

### A — fine-tune from G7-base (recommended: faster, and it's what worked before)

```bash
python3 -u train_interim.py \
    --data ../G7_DATASET/PROCESSED/g7_finetune_ms_snsd_fullrange \
    --init runs/g7_hop256_50k/best.pt \
    --epochs 24 --batch 32 --lr 5e-4 \
    --hop 256 --ch 32,48,48,64 --no-phase --fullband --df \
    --out runs/g7_finetune_fullrange
```

`--init` loads G7-base's weights first; the deep-filter head (not present in
G7-base) initialises fresh and starts at identity, so it doesn't fight the rest
of the network — same trick that made the deep-filter stage trainable at all in
`docs/06-L1-artefact-evidence.md`.

*Caveat, stated plainly:* if your rejected 0dB run was trained from scratch
(no `--init`), this isn't a perfectly isolated single-variable comparison
against it — the initialisation differs too. It **is** a clean answer to the
question that actually matters: does full-range data with these noise sources
help. If you want the stricter comparison, run B as well.

### B — from scratch (stricter isolation vs your exact 0dB run, slower)

```bash
python3 -u train_interim.py \
    --data ../G7_DATASET/PROCESSED/g7_finetune_ms_snsd_fullrange \
    --epochs 24 --batch 32 --lr 2e-3 \
    --hop 256 --ch 32,48,48,64 --no-phase --fullband --df \
    --out runs/g7_finetune_fullrange_scratch
```

**The check that matters, 30 seconds after start:** must print
`model 51,839 params` and `hop 256`. If you see `49,663`, the `--df` flag
didn't apply — stop, you're training G7-base again by accident.

## Evaluate — against OUR sets, not your 0dB test set

This is the step that makes the result mean something. Your dataset's manifest
has train/val rows only, no test split — evaluating against it isn't
comparable to anything else in this project.

```bash
cd E03
for S in edef realnoise; do
  python3 eval_stratified.py --data ../handoff/data/$S \
      --ckpt runs/g7_finetune_fullrange/best.pt \
      --out runs/g012/eval --tag g7ft2_on_$S
done
```

If you don't have `../handoff/data/edef` and `../handoff/data/realnoise` — ask
and I'll get them to you; they're the same 300+300 held-out sets everything
else in this project is scored on.

## The decision rule — apply it yourself before calling it a win

Keep only if, on **both** sets:
- PESQ goes up
- STOI does not go down
- SI-SAR does not go down

vs G7-base (`runs/g7_hop256_50k/best.pt`, evaluated: E_def 0.8151/1.6351/10.05/16.47/11.91,
realnoise 0.8428/1.6128/9.61/16.01/11.65 — STOI/PESQ/SI-SDR/SI-SIR/SI-SAR).

A gain in SI-SIR that costs SI-SAR is not a win, even if SI-SDR looks flat or
slightly up — that was the exact trap in the 0dB run.

## Send back

```
runs/g7_finetune_fullrange/best.pt
runs/g7_finetune_fullrange/last.pt
runs/g7_finetune_fullrange/history.json
runs/g012/eval/eval_g7ft2_on_edef.json
runs/g012/eval/eval_g7ft2_on_realnoise.json
```

If running `eval_stratified.py` yourself isn't convenient, send just `best.pt`
and I'll evaluate it exactly as I did the first one.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
