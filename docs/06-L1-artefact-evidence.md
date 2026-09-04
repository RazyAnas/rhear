# L1 artefact investigation — H9 / H10 evidence

**Frozen.** These are negative and diagnostic results on the L1 speech enhancer,
recorded so they are not re-run and not quietly forgotten. All measured on the
frozen H3 checkpoint (`runs/h9_frozen/h3_snapshot.pt`,
sha256[:16] `53cde5a772f9d636` — byte-identical to `runs/h3_20k/best.pt`).

---

## The finding these all serve

On E_def, 300 clips:

| | value | meaning |
|---|---|---|
| SI-SIR | **20.18 dB** | noise removed — already past the 15 dB requirement |
| SI-SAR | **10.50 dB** | damage caused — this is what caps everything |
| SI-SDR | 9.59 dB | net, limited by SAR not SIR |
| SI-SAR ceiling on our own grid | 22.91 dB | 12.4 dB of headroom |

**RHEAR does not have a noise-suppression problem. It has a speech-damage
problem.** This is only visible because SDR is split into SIR and SAR; the single
SDR number hides it completely.

---

## H9 — comb / harmonic post-filter: REJECTED

*Hypothesis:* the PESQ deficit comes from noise surviving between voice
harmonics, which a 48-band mask is too coarse to remove. A pitch-aligned comb
filter has resolution set by the pitch period rather than the band grid.

| | E_def (300) | E_env (300) |
|---|---|---|
| dPESQ | **+0.004** | +0.014 |
| dSTOI | −0.0017 | −0.0009 |
| dSI-SAR | −0.27 | −0.18 |

Fired on 46.2% of frames at 1.2% of real time, so it had every chance.
Content-preservation check: **0 of 77,177,100 bins** exceeded the input on each
set.

**Rejected**, and the null result is what identified the real problem: after a
mask whose median gain is 0.090, there is almost no inter-harmonic noise left to
remove. The hypothesis was aimed at residue; the problem is distortion.

---

## SAR ceiling probe — it is a TRAINING problem, not a representation one

Oracle masks, ideal ratio clipped to ≤ 1 (what `tanh` permits), 60 clips:

| family | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| A noisy | 0.754 | 1.263 | 3.00 | 3.00 | — |
| B ideal @48 bands, noisy phase | 0.955 | 2.838 | 13.99 | 20.58 | 15.43 |
| **C ideal @48 bands, clean phase** | 0.988 | 4.113 | 22.70 | 37.81 | **22.91** |
| D ideal @257 bins, clean phase | 0.996 | 4.520 | 26.59 | 41.23 | 26.82 |
| E H3 as trained | 0.775 | 1.506 | 9.68 | 19.01 | 10.77 |

**12.15 dB of SI-SAR headroom on our own 48-band grid.** The representation is
not the limit — the model is not finding what the grid already allows.

Note row B: a perfect magnitude mask at 48 bands with the **noisy phase left
untouched** already gives PESQ 2.838, above the 2.5 target. Neither more bands
nor phase prediction is required to reach the target; a better magnitude mask is.

---

## Phase-branch isolation — the branch is HARMFUL

Same magnitude mask, only the phase source varied. 300 clips per set.

| phase source | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| **E_def** predicted (ships today) | 0.788 | 1.529 | 9.59 | 20.18 | 10.50 |
| **E_def** zero (branch off) | **0.811** | **1.570** | 9.65 | 14.84 | **12.11** |
| **E_def** oracle (clean) | 0.847 | 1.826 | 13.58 | 31.67 | 13.78 |
| **E_env** predicted | 0.813 | 1.510 | 8.95 | 19.98 | 10.15 |
| **E_env** zero | **0.835** | **1.548** | 9.13 | 14.46 | **11.80** |

**Switching the branch off improves STOI, PESQ and SI-SAR on both sets**
(+0.023/+0.022, +0.041/+0.038, +1.62/+1.65 dB). Mean predicted rotation ≈ 35°.

*Mechanism.* `mph = atan2(mi, mr)` is computed after `erb.inverse` spreads a
48-band output across 257 bins, so **one rotation is applied to every bin in a
band**. Phase is circular and wraps quickly across frequency, and our widest
band spans 19 bins — a band-constant rotation is ill-posed there.

Removing it also drops `Atan`: unsupported operators **12 → 7**.

**This is an inference-time ablation.** Whether the gain survives retraining is
the open question the phase-free run answers.

---

## H10 — inference-time mask post-processing: REJECTED

*Hypothesis:* SI-SIR is 20.2 dB against a 15 dB requirement, so ~5 dB of
suppression can be traded for quality. Soften the mask and recover SAR.

Thirteen settings on frozen weights, 100 E_def clips. Every one **lowered**
SI-SAR.

| setting | dSTOI | dPESQ | dSI-SAR | dSI-SIR |
|---|---|---|---|---|
| floor f=0.02 | +0.001 | +0.004 | −0.06 | +0.19 |
| floor f=0.05 | −0.000 | +0.003 | −0.19 | +1.56 |
| floor f=0.15 | −0.012 | −0.013 | −0.80 | +3.81 |
| power g=0.8 | −0.007 | −0.010 | −0.13 | +0.02 |
| power g=0.5 | −0.026 | −0.051 | −0.77 | +0.24 |
| time-smooth α=0.30 | −0.009 | −0.013 | −0.18 | +0.68 |
| time-smooth α=0.85 | −0.044 | −0.039 | −1.61 | +1.78 |
| smooth 0.7 + floor 0.02 | −0.025 | −0.024 | −0.75 | +1.73 |

Best PESQ change: **+0.004** — noise.

**Rejected.** And informative in a way the hypothesis did not predict: softening
the mask *raised* SIR (19.50 → 23.31) while *lowering* SAR. You cannot buy
quality back by suppressing less, and temporal smoothing — the standard remedy
for musical noise — did not help either.

**The artefacts are in the trained weights, not in how the mask is applied.** No
inference-time transform fixes this.

*On a prior result.* Gain floors were tested earlier and recorded as harmful,
judged on SI-SDR and STOI. This re-run asked the question those metrics could not
answer, and reached the same verdict for a clearer reason.

---

## G3 — structured pruning: REJECTED, for two independent reasons

*The PS names "quantization, pruning, and ONNX." We had the other two.*

**Reason 1 — there is no redundancy to remove.** Structured masking of the
least-important GRU hidden units on the frozen checkpoint, 100 E_def clips, no
fine-tuning:

| units masked | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| 0% (0 of 80) | 0.747 | 1.465 | 8.32 | 19.50 | 9.26 |
| **5% (4)** | **0.704** | **1.406** | 6.82 | 25.28 | **7.18** |
| 10% (7) | 0.683 | 1.363 | 5.73 | 24.04 | 6.22 |
| 20% (16) | 0.691 | 1.306 | 5.28 | 19.38 | 6.48 |
| 30% (24) | 0.542 | 1.139 | −3.53 | 24.97 | −1.84 |

Removing **4 of 80 hidden units** costs STOI −0.043, PESQ −0.059 and SI-SAR
−2.08 dB. Quality falls off immediately; there is no slack.

Note the familiar pattern: SI-SIR *rises* to 25.3 while everything else
collapses. Damaging the model makes it suppress harder and damage more — the
same SIR/SAR trade seen in H10.

**Reason 2 — even if it worked, it would buy nothing we need.** GRU is 42.5% of
L1's 58.79 MMAC/s, so a successful 20% GRU prune saves **5.0 MMAC/s on a core
with 119 to spare**, and 2.2 KB of Flash out of 1 MB. Activation RAM is 4.59 KB
of 192 KB. The embedded bottlenecks are **operator support** (7 unsupported) and
**sequential GRU steps**, and pruning addresses neither.

*Caveat, stated properly.* Standard practice is prune-then-fine-tune, and this
test masks without fine-tuning, so a fine-tune might recover part of the loss.
That does not change the verdict, because Reason 2 stands on its own: the saving
is negligible against the budget we actually have. Worth revisiting only if the
model grows enough that compute becomes binding.

*Why the model has no redundancy:* at 22,956 parameters it is already half of
GTCRN's 48.2 k, the efficiency reference it was scaled against. It was designed
small; there is nothing left to squeeze.

---

## G0 — phase-free retrain: CONFIRMED, and it stabilised training

> **Attribution: RESOLVED.** This curve was briefly reported as G1. The probe
> below predicted from its loss level alone that it contained no perceptual
> term, and the checkpoints confirmed it: all three loaded into the model with
> zero missing or unexpected keys at **22,956 / 22,956 / 24,975** trainable
> parameters, `phase=False` throughout and `fullband=True` on G2 only. G1 did
> run correctly — its epoch-0 loss of **+11.17** is the perceptual signature.
> This section is G0.

The phase-branch isolation above switched the branch off at inference on an
already-trained model. G0 is the retrain: the branch is gone from the start, so
capacity that was being spent on it is available to the magnitude path. One
variable — `--no-phase`. Same data, splits, optimizer and 24-epoch budget as the
frozen H3 baseline, which makes the two curves directly comparable epoch by
epoch.

| | H3 (with phase) | G0 (no phase) | |
|---|---|---|---|
| best val SI-SDR | 9.449 dB | **9.821 dB** | **+0.372** |
| final val SI-SDR | 9.402 dB | **9.821 dB** | **+0.418** |
| epoch of best | 19 | **23** | still rising at the cut |
| spread over last 6 epochs | 0.279 dB | **0.022 dB** | 13x steadier |
| non-finite batches skipped | 0 | 0 | |

G0 is ahead at **every epoch from 2 onward**, by +0.19 to +0.63 dB. This is not
a lucky final epoch.

*Train loss is not comparable between the two* — the phase variant's objective
carries real/imaginary terms the phase-free one does not, so the two numbers
measure different quantities. `val_sisdr` is the comparable column, which is why
the table reports only that.

**The second result is the more interesting one.** H3's validation curve
oscillates: it peaks at epoch 19, falls, and never recovers, with a 0.279 dB
spread across the final six epochs. G0's final six sit inside 0.022 dB. Removing
the branch did not merely raise the score, it removed most of the run-to-run
noise in it.

That is consistent with the mechanism already diagnosed: `atan2` is computed
after 48 bands are spread back over 257 bins, so one predicted angle is applied
to a band up to 19 bins wide, while phase wraps faster than that. A gradient
computed against that target is close to noise. Training against it perturbed
the shared encoder every step — which is what the oscillation was.

It also removes the reason the pending val-SI-SDR mean-fragility fix looked
urgent. That fragility is largely an artefact of the branch that is now gone.

**Not yet established, and not to be claimed:** whether the +1.62 dB SI-SAR,
+0.023 STOI and +0.041 PESQ from the inference-time isolation survive
retraining. Validation SI-SDR is a different quantity on a different split from
the E_def and realnoise evaluations. That question is answered by
`E03/run_after_retrain.sh <ckpt>` and needs `best.pt`, which has not arrived.

**Converged — do not extend.** 0.022 dB across six epochs is a plateau; more
epochs on this configuration buy nothing. The GPU should move to G1.

Curve: `E03/runs/g0_nophase/history.json`.

---

## Which variant produced the 24-epoch curve — and why `--perceptual 1.0` is too strong

`history.json` records only `epoch`, `train_loss`, `val_sisdr` and `skipped`.
The `perc` value is printed to stdout only, so the file does not state which
variant wrote it. That had to be measured.

`--perceptual` does not change the model — G0 and G1 are both 22,956 parameters,
so **the parameter-count check in the handoff README cannot catch a dropped
`--perceptual` flag.** That is a gap in the handoff instructions, not in the
experiment. What it does change is the objective: a non-negative term is added,
so the two variants sit at different loss levels on identical batches.

Probe: train the plain `--no-phase` model on the real H3 data and, at every
step, evaluate the perceptual term as well without letting it touch the
gradient — giving both variants' recorded loss from one run.

| after 150 steps | mean train_loss |
|---|---|
| G0 (`--no-phase`) | **+1.94**, still falling steeply |
| perceptual term alone | **+14.41** |
| G1 (`--no-phase --perceptual 1.0`) would record | **+16.36** |
| **observed epoch 0** | **−2.10** |

The observed run sits on the G0 trajectory — a running mean of +1.94 at step 150
and still dropping reaches roughly −2 over a full 625-step epoch. A G1 run would
have recorded **about +16 at epoch 0**. The gap is an order of magnitude wider
than the estimate's error.

**G1 is ruled out for this curve. G0 and G2 are not separable from it** — they
share the same loss composition and differ only in parameter count.

### The more useful finding: weight 1.0 makes the perceptual term dominate

The base objective falls from +10.3 to below zero over 150 steps while the
perceptual term **stays near 14** (11.4 → 17.1 → 15.0 → 14.4 → 13.5 → 14.4). At
weight 1.0 it is therefore not a refinement on top of the objective — it becomes
roughly **seven times the base loss** and effectively the whole objective,
drowning the SI-SDR term and the asymmetric magnitude term that produce our
noise suppression in the first place.

This predicts exactly what Suryansh observed: **a G1 run scoring 8.77 dB val
SI-SDR against 9.82** is the expected consequence of that weighting, not a
broken run.

**A val-SI-SDR drop does not reject G1.** Perceptual loss is meant to trade
SI-SDR for perceptual quality, and PESQ — 1.56 against a 2.5 target — is the
metric it exists to move. Judging it on val SI-SDR measures the one thing it was
never intended to improve. G1 is decided by `E03/run_after_retrain.sh` on PESQ,
STOI and SI-SAR over E_def and realnoise, under the standing rule.

If PESQ does not move at weight 1.0, the next step is **weight 0.1–0.2**, not
abandoning the term: the measurement above says 1.0 was chosen too high.

Probe: `scratchpad/probe_variant.py`.

---

## G1 / G2 evaluated — G1 REJECTED, G2 KEPT

All three checkpoints arrived and were verified before use: loaded into the
model with **zero missing and zero unexpected keys**, at 22,956 / 22,956 /
24,975 trainable parameters, `phase=False` throughout and `fullband=True` on G2
only. G1's epoch-0 train_loss of **+11.17** confirms `--perceptual` applied, as
the probe above predicted. Evaluation is 300 E_def and 300 realnoise clips.

### Overall

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| **E_def** | | | | | |
| H3 (phase) | 0.788 | 1.529 | 9.59 | **20.18** | 10.50 |
| G0 (no-phase) | 0.810 | 1.604 | 9.92 | 16.28 | 11.83 |
| G1 (+perceptual) | 0.792 | 1.493 | 8.89 | 14.87 | 11.12 |
| **G2 (+full-band)** | **0.814** | **1.629** | **10.05** | 16.40 | **11.89** |
| **realnoise** | | | | | |
| H3 (phase) | 0.813 | 1.510 | 8.95 | **19.97** | 10.15 |
| G0 (no-phase) | 0.836 | 1.568 | 9.17 | 15.10 | 11.55 |
| G1 (+perceptual) | 0.814 | 1.438 | 7.62 | 12.64 | 10.87 |
| **G2 (+full-band)** | **0.839** | **1.593** | **9.35** | 15.37 | **11.61** |

### G1 — perceptual loss at weight 1.0: REJECTED

Worse than G0 on **every metric on both sets**: PESQ −0.112 / −0.131, STOI
−0.019 / −0.022, SI-SAR −0.72 / −0.69, SI-SIR −1.40 / −2.46. It failed at the
one thing it was added to do.

This is the predicted consequence of the weighting, not a broken run. The probe
above measured the perceptual term at ~14 while the base objective falls below
zero — at weight 1.0 it is roughly **seven times** the base loss and effectively
replaces the objective. Its validation curve also collapsed to **3.94 dB at
epoch 9** before recovering, the worst instability of any run in this series.

*Not a rejection of perceptual loss as such — a rejection of weight 1.0.* The
open question is **weight 0.1–0.2**, which would put the term at roughly 0.7x
the base instead of 7x. That is a single-variable experiment and it is the one
worth running next, because PESQ (1.63 against a 2.5 target) remains our worst
metric by a wide margin.

### G2 — full-band branch: KEPT

Better than G0 on **all five metrics on both sets**, and better in **66 of 75**
stratum-metric cells, including PESQ in **8 of 8 defence noise classes**, both
reverb conditions and 11 of 12 SNR buckets.

*State the size honestly:* the gain is small — PESQ +0.025/+0.024, STOI
+0.004/+0.003, SI-SAR +0.05/+0.05. The strata partition the same 300 clips three
ways and the metrics are correlated, so the sign test's p-value overstates the
case. What the consistency establishes is **direction, not magnitude**: G2 is
not worse anywhere that matters.

It is kept because the cost is near zero and it closes a PS gap:

| | G0 | G2 | |
|---|---|---|---|
| parameters | 22,956 | 24,975 | +8.8% |
| compute | 58.79 MMAC/s | 62.74 MMAC/s | +6.7% |
| core-1 load | 80.8 | 84.8 | ceiling 200 |
| total utilisation | 65% | **66%** | **FRAME DEADLINE PASS** |
| ONNX distinct op types | 18 | **18** | **no new types** |
| not CMSIS-NN native | 7 | **7** | unchanged |

**No `Resize` operator appears.** The (4,4,3) stride choice — picked so 257 bins
land on exactly 6 — did its job; the +19 ONNX nodes all reuse operator types
already present. G2 costs one percentage point of the compute budget and no new
embedded porting work, and it makes the PS's "full-band **and** sub-band"
requirement true rather than argued.

### Did the phase-removal gain survive retraining? YES

| set | dSTOI | dPESQ | dSI-SDR | dSI-SIR | dSI-SAR |
|---|---|---|---|---|---|
| predicted (inference ablation) | +0.023 | +0.041 | — | — | +1.62 |
| **E_def measured** | **+0.023** | **+0.075** | +0.33 | **−3.90** | **+1.34** |
| predicted | +0.022 | +0.038 | — | — | +1.65 |
| **realnoise measured** | **+0.023** | **+0.059** | +0.22 | **−4.87** | **+1.41** |

STOI reproduces exactly. SI-SAR reproduces slightly under prediction. **PESQ came
in above prediction on both sets.** Phase removal is confirmed and stays.

### The cost nobody predicted, and it must not be buried

**SI-SIR fell 3.9–4.9 dB.** H3 suppressed ~20 dB of noise; G2 suppresses 16.4
(E_def) and **15.37 (realnoise)**. The model became gentler: it removes less
noise and does less damage, netting a small SI-SDR gain and a clear perceptual
gain.

This changes what we may claim. The PS asks for **SNR > 15 dB**, and we have
been quoting SI-SIR 20.2 dB as clearing it comfortably. **That margin is now
0.37 dB on realnoise.** Two things follow:

1. Quote **SI-SDR 9.35 dB** as the honest net figure — it does not meet 15 — and
   SI-SIR 15.4 dB as the suppression component, with its margin stated. Do not
   present 15.37 as comfortable.
2. Any future change must now watch SI-SIR as a **constraint**, not a free
   variable. It has been spent.

Against that, the metrics that were failing improved: realnoise STOI is
**0.839 against a 0.85 target — 0.011 short**, from 0.813. That is the closest
this project has come to a PS target it was missing.

Runs: `E03/runs/g{0,1,2}_*/`. Evaluations: `E03/runs/g012/eval/`.

---

## H11 — triangular ERB synthesis at inference: REJECTED

**Hypothesis, from the literature rather than from a sweep.** `ERBSplit.inverse`
spreads each band gain to its bins with a **binary** matrix
(`Minv = (M > 0)`), so the applied gain is piecewise-constant and every band
edge is a step discontinuity — at 48 bands over 257 bins the widest bands are
~19 bins across. PercepNet (Valin 2020) and DeepFilterNet (Schröter 2022) both
use **overlapping triangular** filterbanks instead, and spectral discontinuity is
a known source of the roughness PESQ penalises. Prediction: smooth
band-to-bin interpolation raises PESQ at zero training cost and zero runtime
cost.

Swapped the synthesis for linear interpolation between adjacent band centres in
ERB space. Model, weights and outputs identical; only the spreading differs.

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def rectangular | 0.8141 | 1.6295 | 10.05 | 16.40 | 11.89 |
| E_def triangular | 0.8095 | 1.6571 | 8.94 | 17.07 | 10.26 |
| **delta** | −0.005 | **+0.028** | −1.10 | +0.67 | **−1.63** |
| realnoise rectangular | 0.8393 | 1.5927 | 9.35 | 15.37 | 11.61 |
| realnoise triangular | 0.8355 | 1.6539 | 8.72 | 16.46 | 10.41 |
| **delta** | −0.004 | **+0.061** | −0.63 | +1.09 | **−1.20** |

**The PESQ prediction was correct — it rose on both sets — and the change is
still rejected**, because it fails the other two conditions on both sets. What it
actually does is trade damage for suppression: SI-SIR up 0.67/1.09, SI-SAR down
1.63/1.20. That is the wrong direction on the metric this project has repeatedly
established as binding.

*The confound, stated because it decides what happens next.* G2 was **trained**
with the rectangular operator, so its gains are learned to be applied as boxes.
Swapping the operator only at inference is a train/test mismatch, and this
experiment therefore does not show that triangular synthesis is wrong — only
that it cannot be bolted on for free. Since it costs nothing at runtime (2
non-zeros per bin instead of 1) and its PESQ effect was consistent and in the
predicted direction, it is a reasonable **single variable for a future retrain**,
not a rejected idea.

Probe: `E03/erb_synthesis_test.py`. Results: `E03/runs/g012/erb_synthesis.json`.

---

## G5 — perceptual loss at the corrected weight 0.15: REJECTED (weak test)

G1 failed at weight 1.0, and the probe explained why: the perceptual term sits
near 14 while the base objective goes negative, so weight 1.0 made it roughly
**7x the base loss** and it replaced the objective. The obvious follow-up was the
weight that puts it at ~0.7x instead. Fine-tuned from G2 (4 epochs, lr 3e-4,
`--init` added to `train_interim.py` for this) rather than trained from scratch,
which is what made it affordable at 85 minutes.

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def G2 | 0.8141 | 1.6295 | 10.05 | 16.40 | 11.89 |
| E_def G5 | 0.8134 | 1.6167 | 10.07 | 16.60 | 11.82 |
| **delta** | −0.0007 | **−0.0127** | +0.02 | +0.21 | −0.06 |
| realnoise G2 | 0.8393 | 1.5927 | 9.35 | 15.37 | 11.61 |
| realnoise G5 | 0.8389 | 1.5875 | 9.40 | 15.56 | 11.61 |
| **delta** | −0.0003 | **−0.0053** | +0.06 | +0.19 | +0.00 |

Everything is flat to within noise, and **PESQ — the metric the term exists to
move — went slightly down on both sets.** Rejected.

Two independent points now bracket this avenue: at weight 1.0 the perceptual
term **actively harms** (PESQ −0.11 / −0.13); at 0.15 it does **nothing**. No
useful window between them has shown itself, and the term is not free to search.

**Two honest weaknesses in this test, stated so nobody over-reads the verdict:**

1. `best.pt` was **epoch 0**. Checkpoint selection is on validation SI-SDR, so
   when the intervention deliberately trades SI-SDR for perceptual quality, the
   selector picks the checkpoint with the *least* of the thing being tested. The
   effective dose was one epoch. **The training script should also save
   `last.pt`** — this is a methodological bug that will bias every future
   perceptual or quality-targeted run the same way.
2. Validation SI-SDR was flat across all four epochs (10.026 / 9.984 / 9.986 /
   9.984) while train loss kept improving (−7.976 → −8.175). Whether epoch 3
   would have scored better on PESQ is unknown, because it was not kept.

So this is a **weak negative, not a strong one**. It does not justify a third
attempt at tuning the weight; it justifies fixing the checkpoint selection before
any further quality-targeted training.

Run: `E03/runs/g5_perc015/`.

---

## ORACLE LADDER — how much room each design actually has

The question that matters is not "can we train harder" but "which design has a
ceiling above the target". An oracle cannot be beaten by any model of its
family, so a family whose oracle sits near 2.5 is finished before it starts.
No training involved. `E03/oracle_ladder.py`, 40 clips, deep filter order N=5.

| family | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| A noisy | 0.761 | 1.250 | 2.14 | — | — |
| **E — G2 as trained** | **0.795** | **1.527** | **8.81** | 15.24 | 10.92 |
| B ideal mask @48 bands, noisy phase | 0.956 | **2.731** | 13.18 | 19.63 | 14.71 |
| F ideal mask @257 bins, noisy phase | 0.966 | 3.179 | 14.33 | 21.37 | 15.66 |
| **H ERB + deep-filter cascade (96 ms)** | **0.968** | **3.297** | **17.37** | 29.77 | 17.81 |
| H same, 192 ms | 0.957 | 3.073 | 16.06 | 26.99 | 16.66 |
| C ideal @48, clean phase | 0.987 | 4.070 | 22.20 | 37.39 | 22.44 |
| D ideal @257, clean phase | 0.996 | 4.511 | 25.72 | 40.35 | 25.97 |

**The deep-filter cascade raises the PESQ ceiling from 2.731 to 3.297** — and
the SI-SAR ceiling from 14.71 to 17.81 dB, which matters more, because SI-SAR is
our binding constraint. Resolution alone (F) buys +0.45; the cascade buys +0.57
on top of a coarser grid, confirming the published result that the second stage
matters more than the band count.

### The number that reframes the project

Required efficiency against the 2.5 PESQ target:

| design | ceiling | fraction of oracle needed |
|---|---|---|
| today (ERB only) | 2.731 | **92%** |
| ERB + deep filter | 3.297 | **76%** |
| **we currently achieve** | — | **56%** (1.527 / 2.731) |

**Raising the ceiling is necessary and not sufficient.** At today's 56%
efficiency a 3.297 ceiling yields ~1.85, still short of 2.5. The target needs
**both** a higher ceiling *and* a jump from 56% to 76% of it. Any plan that
addresses only one of those will miss, and that is why the earlier
quick-win attempts (H11 synthesis, G1/G5 perceptual loss) could not have worked:
every one of them was an efficiency play against a ceiling too low to matter.

### Two of our own errors, caught by the ladder

Worth recording because both produced confident, wrong tables.

1. **Time-invariant coefficients.** The first version solved one filter per clip
   and scored PESQ 1.579 against the real mask's 3.275 — impossible, since a
   time-varying real mask is the special case `c = [g,0,0,0,0]`. It was
   measuring "time-invariant vs time-varying", not "complex vs real".
2. **Conjugate on the wrong index.** `R[j,i] = Σ conj(Z[t,j]) Z[t,i]`; the code
   built the transpose and so solved `conj(R) c = p`. Fixed, then unit-checked:
   5 taps must beat 1 tap and both must beat doing nothing.
3. **DF as a replacement rather than a refinement.** Deep filtering is stage 2
   *on top of* the ERB gains, not a substitute below 5 kHz. Fitting stage 2 on
   the already-masked signal is what lets it act as identity where the mask was
   already right — and it moved H from 2.275 to 3.297.

*A caveat that must travel with this table:* the oracle is sensitive to how fast
the coefficients may vary. At a 32 ms window the fit is near-degenerate (8
equations for 5 unknowns) and reports an implausible 4.276 — **discard that
row.** The 96 ms and 192 ms figures are the defensible ones, and the 192 ms
number is the conservative one to quote.

---

## G6 — deep-filtering cascade: BUILT, WORKS, BADLY UNDERTRAINED

The stage the oracle ladder pointed at. 5 complex taps per bin below 5.7 kHz,
applied across time to the ERB stage's output, residual around identity.
26,479 params (+1,504), 97.3 MMAC/s (+34.6), core 1 at 119.3 of 200, chip 74%,
frame deadline PASS, and **no new ONNX operator types** — the set is identical
to G2's seven.

Fine-tuned from G2 for 5 epochs. Validation SI-SDR rose monotonically
9.929 → 10.114, above G2's 10.038 and **still climbing at the cut**. Zero
non-finite batches.

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def G2 | 0.8141 | 1.6295 | 10.05 | 16.40 | 11.89 |
| E_def G6 | 0.8155 | 1.6416 | 10.15 | 16.77 | 11.86 |
| **delta** | +0.0014 | **+0.0121** | +0.10 | +0.37 | **−0.03** |
| realnoise G2 | 0.8393 | 1.5927 | 9.35 | 15.37 | 11.61 |
| realnoise G6 | 0.8409 | 1.6141 | 9.50 | 15.64 | 11.67 |
| **delta** | +0.0016 | **+0.0214** | +0.16 | +0.27 | +0.06 |

**Under the standing rule this is a REJECT**, on one cell: SI-SAR −0.03 dB on
E_def. Every other cell improves on both sets — 9 of 10 positive.

*But do not read that as a refutation, and do not read it as a pass either.*
−0.03 dB is indistinguishable from noise, and calling it a rejection is as
misleading as calling the +0.012 PESQ a win. The honest summary is that the
cascade moved everything very slightly in the right direction.

### The finding that actually matters

| | |
|---|---|
| headroom the cascade opened (ceiling 2.731 → 3.297) | **+0.566 PESQ** |
| headroom the model captured | **+0.012 to +0.021** |
| **fraction of the new headroom realised** | **~3%** |

And the efficiency figure moved the wrong way:

| | fraction of its own ceiling achieved |
|---|---|
| G2 against the ERB ceiling (2.731) | **60%** |
| G6 against the cascade ceiling (3.297) | **50%** |

**Raising the ceiling did not help, because we were never near the ceiling.**
This is the clearest statement yet of where the project actually stands: the
binding constraint is not the representation, it is how little of the
representation the training extracts. That was already visible — G2 at 60% of
2.731 — and G6 confirms it by moving the ceiling 21% and the result 1%.

### Why this run is not the verdict on deep filtering

Five epochs of fine-tuning is roughly 3,100 steps for a head that started at
identity and had to learn 1,830 coefficients per frame from scratch.
DeepFilterNet trains this stage for hundreds of epochs. The signals all point at
undertraining rather than a dead end:

- validation SI-SDR was **still rising at the last epoch** (10.090 → 10.114)
- train loss still falling (−8.789 → −9.000)
- every metric moved in the right direction on both sets

**The correct next step is a full-length run from scratch on real hardware, not
another laptop fine-tune.** Until that exists, G6 is neither kept nor rejected —
it is unmeasured at the dose that matters.

Runs: `E03/runs/g6_deepfilter2/`. Model flag `--df`. Oracle: `E03/oracle_ladder.py`.

---

## CROSS-BENCHMARK — we are not behind. The test set is brutal.

The most important measurement in this project, and it took no training.

We had been reading "PESQ 1.63 against a 2.5 target" as a model failure. That
reading was never tested against a published model on the same data. So: take
**GTCRN** (Rong et al., ICASSP 2024) — 48,245 parameters, **1.93x our size**,
peer-reviewed, with **PESQ 2.87 / STOI 0.940 published on VoiceBank+DEMAND** —
run its own released weights through its own published inference path
(16 kHz, n_fft 512, hop 256, sqrt-Hann) over **our** 300 held-out defence clips,
scored by **our** scorer. Nothing differs but the model.

| model | params | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| (noisy input) | — | 0.7688 | 1.2915 | 2.50 | — | — |
| GTCRN, trained on VCTK-DEMAND | 48,245 | 0.7648 | 1.4442 | 6.45 | 12.19 | 9.70 |
| GTCRN, trained on DNS3 | 48,245 | 0.8182 | 1.6430 | 9.61 | 18.32 | 10.79 |
| **RHEAR G2** | **24,975** | 0.8141 | 1.6295 | 10.05 | 16.40 | **11.89** |
| **RHEAR G6** | **26,479** | 0.8155 | **1.6416** | **10.15** | 16.77 | 11.86 |

**The same GTCRN weights that score 2.87 on VoiceBank+DEMAND score 1.643 on our
data — a loss of 1.23 PESQ from changing nothing but the test set.**

Against their stronger checkpoint we are level: dPESQ −0.001, dSTOI −0.003,
**dSI-SDR +0.53, dSI-SAR +1.07**, with **half the parameters**. Their
VCTK-trained checkpoint is far worse than ours and actually *lowers* STOI
(0.7688 → 0.7648) on defence noise.

### What this settles

1. **Our model is not the problem.** A published ICASSP model at 1.93x our size
   lands within 0.001 PESQ of us on our own data. The "we only capture 60% of
   the oracle" framing was real but misleading — a strong published model
   captures no more of it than we do.
2. **The 2.5 PESQ target is calibrated on far easier material.** Benchmarks like
   VoiceBank+DEMAND test at +2.5 to +17.5 dB SNR against domestic noise, and
   their noisy input starts at PESQ ~1.97. Ours starts at **1.29**, with a third
   of clips below 0 dB and noise that includes gunshots and artillery.
3. **The last four experiments were aimed at the wrong target.** H11, G1, G5 and
   arguably G6 all tried to close a gap that is mostly not ours to close.

### What it does NOT excuse

The PS states the numbers plainly and a panel will hold us to them. The correct
response is not "our test is harder, so score us differently" — it is to report
**both**: our number on our defence set, and this cross-benchmark showing what a
state-of-the-art model of twice our size does on the same clips. That is a far
stronger position than a bare 1.64, and it is reproducible from public weights.

Reproduce: `E03/crossbench.py --ckpt dns3|vctk`. Weights from
https://github.com/Xiaobin-Rong/gtcrn (`checkpoints/`).

### The capacity headroom this reveals

GTCRN carries 48,245 parameters at a **published ~40 MMAC/s** while we spend
**62.7 MMAC/s on 24,975**. The difference is not efficiency, it is frame rate:
our hop is 64 samples (**250 frames/s**), theirs is 256 (**62.5 frames/s**). We
run our network **four times more often than a comparable published system**,
and are spending the compute budget on temporal oversampling rather than on
capacity.

| | hop | frames/s | MMAC/s | params |
|---|---|---|---|---|
| RHEAR L1 today | 64 | 250 | 62.7 | 24,975 |
| GTCRN | 256 | 62.5 | ~40 | 48,245 |
| **RHEAR at hop 256** | 256 | 62.5 | **~15.7** | 24,975 |

Moving L1 to hop 256 frees roughly **47 MMAC/s** — enough for **3-4x the
parameters inside the existing budget**. The cost is algorithmic latency on the
comms path only: 4 ms frames become 16 ms. **L0's 146 us causality budget is
untouched** — that is a different path and a different chip constraint. 16-32 ms
is ordinary for voice communication; standard telecom codecs use 20 ms frames.

---

## G7 — hop 256 + 50k params: KEPT. Same quality, half the compute.

The user's plan: keep the architecture, move L1's temporal operating point to
the one GTCRN proves works, and spend the freed compute on capacity.

| | G2 | **G7** |
|---|---|---|
| hop | 64 (250 frames/s) | **256 (62.5 frames/s)** |
| params | 24,975 | **49,663** |
| compute | 62.74 MMAC/s | **32.34 MMAC/s** |
| core 1 | 84.8 / 200 | **54.3 / 200** |
| training | ~25 min/epoch | **4.7 min/epoch** |

24 epochs from scratch, 113.7 min, zero non-finite batches, best val SI-SDR
9.95 (G2: 10.04).

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def G2 | 0.8141 | 1.6295 | 10.05 | 16.40 | 11.89 |
| E_def **G7** | **0.8151** | **1.6351** | 10.05 | 16.47 | **11.91** |
| **delta** | +0.0010 | +0.0056 | +0.00 | +0.07 | +0.03 |
| realnoise G2 | 0.8393 | 1.5927 | 9.35 | 15.37 | 11.61 |
| realnoise **G7** | **0.8428** | **1.6128** | **9.61** | **16.01** | **11.65** |
| **delta** | +0.0036 | +0.0201 | +0.26 | +0.63 | +0.04 |

**Every metric improves on both sets. PASS on the standing rule** — the first
clean KEEP in this series, after H11, G1, G5 and G6 all failed it.

*State the size honestly:* the quality gain is small (+0.006 / +0.020 PESQ).
**The result is not the quality gain — it is the cost.** We get slightly better
numbers for **half the inference compute**, core 1 drops from 42% to 27% of its
budget, and experiments now cost under two hours instead of overnight.

### Why it worked

Not capacity — 2x the parameters bought almost nothing on its own, consistent
with everything else here. The win is that we had been running the network
**four times more often than any comparable published system**. GTCRN carries
48,245 params at ~40 MMAC/s; we were spending 62.74 on 24,975 purely because our
hop was 64. That compute was buying temporal oversampling nobody else pays for.

**Cost:** L1's algorithmic latency goes from 4 ms frames to 16 ms. This affects
the COMMS path only — **L0's 146 us causality budget is untouched**, being a
different path on a different core. 16-32 ms is ordinary for voice
communication; standard telecom codecs use 20 ms frames.

### What this unlocks

Core 1 now sits at 54.3 of 200. There is room for **75-100k parameters**, or for
a properly-trained deep-filter stage (G6 cost +34.6 MMAC/s and would now fit
comfortably), or both. The user's instruction was to test ~50k first and go
higher only if it still shows headroom — G7's validation curve was **still
rising at epoch 23**, so it does.

Run: `E03/runs/g7_hop256_50k/`. Flags: `--hop 256 --ch 32,48,48,64 --no-phase --fullband`.

---

## G7-finetune (Suryansh, hop256+50k+deep-filter, MS-SNSD/MAD/DEMAND @ 0dB): REJECTED

Suryansh combined three changes at once from his own GPU: G7's hop256+50k
capacity, the G6 deep-filter cascade, and a retrain on new noise sources
(MS-SNSD, MAD, DEMAND) fixed at exactly 0 dB SNR. His own eval script reports
STOI and SI-SDR only -- no PESQ, no SI-SIR/SI-SAR. Downloaded the real
checkpoint (342,633 bytes, verified against the LFS pointer's declared size)
and ran our full `eval_stratified.py` on our standard E_def/realnoise sets so
it is directly comparable to every other checkpoint in this document.

51,839 params (vs G7-base's 49,663 -- the deep-filter head accounts for the
difference), hop 256, phase removed, full-band present, deep-filter present.

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def G7-base | 0.8151 | 1.6351 | 10.05 | 16.47 | 11.91 |
| E_def G7-finetune | 0.7927 | 1.5257 | 8.22 | 16.91 | 9.64 |
| **delta** | -0.0224 | **-0.1094** | -1.83 | +0.45 | **-2.27** |
| realnoise G7-base | 0.8428 | 1.6128 | 9.61 | 16.01 | 11.65 |
| realnoise G7-finetune | 0.8301 | 1.5632 | 9.14 | 17.06 | 10.73 |
| **delta** | -0.0127 | **-0.0496** | -0.46 | +1.05 | **-0.92** |

**REJECT on the standing rule, both sets.** This is the SI-SIR/SI-SAR trade-off
this project has hit before (H11): SI-SIR is UP on both sets (+0.45, +1.05 --
the model suppresses more), but SI-SAR is down hard (-2.27, -0.92 -- it damages
more), and the net is worse everywhere that matters: PESQ -0.11/-0.05, STOI
down on both sets, SI-SDR down 1.8/0.5 dB.

**Three variables changed at once, so the cause cannot be isolated from this
run alone:**

1. hop256+50k capacity -- already independently verified to HELP (G7-base
   passed the standing rule cleanly against G2)
2. the deep-filter cascade -- previously INCONCLUSIVE at 5 epochs (G6):
   9/10 cells positive but undertrained, val SI-SDR still rising at the cut
3. new training data -- MS-SNSD/MAD/DEMAND, but fixed at EXACTLY 0 dB SNR,
   narrower than H3's -10..+20 dB range and skewed toward the hardest end

Given (1) is already known-good, the regression is in (2), (3), or their
interaction. The single-SNR training distribution is the most likely
suspect: training only ever seeing 0 dB may have taught the model that
aggressive suppression is always correct, which is exactly the SIR-up/SAR-down
signature measured here, and it would explain why the SAME deep-filter head
that moved 9/10 cells positive in G6 (trained on the -10..+20 dB H3 range)
now hurts.

**Do not deploy this checkpoint.** G7-base remains the current best.

**The recommended next step is one variable, not three:** retrain hop256+50k
+deep-filter on the ORIGINAL H3 -10..+20 dB range with the new noise sources
mixed in (MS-SNSD/MAD/DEMAND replacing or supplementing the existing pool,
same SNR distribution), so the new-data question is answered without also
re-asking the already-answered deep-filter and capacity questions.

Checkpoint: `E03/runs/g7_finetune_ms_snsd_0db_v2/best.pt`. Downloaded from the
GitHub PR (`SuryanshSinha2801/g7-integration`), evaluated locally.

---

## Standing decision

Do **not** run further mask-floor, power-compression or temporal-smoothing
sweeps unless a new hypothesis specifically predicts a measurable benefit.

The next intervention targets one of: **training loss**, **perceptual
optimisation**, **model capacity**, or **representation** — chosen only after the
phase-free retrain result, and **one variable at a time**.

---

## G7 full-range rerun — the fix worked, the checkpoint still loses

`SuryanshSinha2801/rhear-g7-pr` @ `ae5a549`, 24 epochs on RTX 5060, 12.8 min.
Same architecture as the rejected 0dB run (51,839 params, hop 256, df=True),
same 70/30 MAD+DEMAND/MS-SNSD mix, SNR range corrected to −10..+20 dB.

| E_def | STOI | PESQ | ΔSI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| 0dB run − G7-base | −0.022 | −0.109 | — | +0.45 | **−2.27** |
| full-range − G7-base | −0.011 | −0.053 | +0.36 | +0.12 | **−0.74** |

**The SNR-range hypothesis is confirmed.** Every symptom of the 0dB run moved
toward zero: the SI-SAR loss shrank from −2.27 to −0.74 dB and the PESQ loss
halved. Single-point SNR training was the cause, as predicted.

**The checkpoint still fails the standing rule.** SI-SIR is still up and SI-SAR
still down — the same signature, weaker. G7-base remains current best. The
residual is now attributable to the deep-filter head, the one variable the
full-range run did not remove, and rows G/Gb below explain why that head cannot
help on the metrics we are graded on.

---

## The PS targets, measured at the SNR the PS actually specifies

`E03/oracle_at_0db.py` · `runs/g012/oracle_at_0db.json`,
`oracle_at_0db_realnoise.json`

**Every number this project has ever reported is a mean over a −10..+20 dB
mix. Every PS acceptance target in `02-architecture.md` §10 is written "at 0 dB
input SNR".** Those are not the same quantity and we had never measured the
second one. This runs the oracle rows on the 54 held-out clips whose manifest
`snr_db` is within ±2.5 dB of zero. No checkpoint is loaded — these are
oracles, so no model of that family can beat them.

| row | STOI | PESQ | ΔSI-SDR |
|---|---|---|---|
| **PS target** | **≥ 0.85** | **≥ 2.5** | **≥ 15** |
| A noisy | 0.657 | 1.079 | 0.00 |
| **B ideal mask @48 ERB, noisy phase** ← our design | **0.938** | **2.392** | **13.02** |
| F ideal mask @257 bins, noisy phase | 0.953 | 2.899 | 14.05 |
| C ideal @48 + CLEAN phase | 0.983 | 3.946 | 22.63 |
| D ideal @257 + CLEAN phase | 0.995 | 4.480 | 26.90 |

E_def shown; realnoise agrees within 0.06 PESQ and 0.05 dB ΔSI-SDR on row B
(0.956 / 2.446 / 12.97).

**Three conclusions, and they are architectural, not about training:**

1. **STOI ≥ 0.85 is the only target inside our current design.** Ceiling 0.938,
   G7-base sits at 0.75 (E_def) / 0.81 (realnoise) around 0 dB. Reachable.
2. **PESQ ≥ 2.5 is unreachable with a 48-band ERB mask.** A *perfect* mask
   scores 2.392. The binding constraint is frequency resolution, and lifting it
   alone crosses the line (row F, 2.899).
3. **ΔSI-SDR ≥ 15 dB is unreachable by any magnitude-only method.** Even a
   perfect 257-bin mask reaches 14.05. Only phase crosses it: clean phase is
   worth **+9.61 dB** of ΔSI-SDR at 48 bands, where resolution is worth +1.04.

**This retires the deep-filter direction for these targets.** On the full-mix
ladder the deep-filter oracle scores PESQ 2.256 / STOI 0.895 against the plain
ideal mask's 2.838 / 0.955 — it buys SI-SIR (28.2 vs 20.6) at the cost of the
two metrics we are graded on. It is the wrong lever, and no amount of training
changes an oracle.

**Correction to an earlier reading:** the "deep filter raises the ceiling to
3.3" note was a misreading of row **F** (ideal@257, 3.275), not row G.

**Consequences for what to build next**, in the order the ceilings justify:

- **Phase, not capacity.** G7 runs `phase=False`. That was decided on overall
  PESQ, which is the metric phase helps least; ΔSI-SDR at 0 dB is where the
  +9.6 dB sits. Re-open the phase branch and judge it on ΔSI-SDR at 0 dB.
- **Frequency resolution.** 48 ERB bands is what caps PESQ. More bands, or a
  hybrid — fine bins across 300 Hz–3.4 kHz, ERB elsewhere.
- **STOI is a training problem.** It is the one target bankable without an
  architecture change.

**Report both conditions from now on.** A mix-average PESQ flatters us relative
to the 0 dB target; a mix-average ΔSI-SDR understates us. Quoting one number
against a target defined at a point is a category error in both directions.

---

## Which plan can clear the PS targets — priced, then chosen

`E03/plan_ladder_0db.py`, `E03/model_at_0db.py`, `E03/plan_demand.py`,
`E03/mac_by_bands.py` · 54 clips at 0 dB ±2.5 dB, both eval sets.

### What the oracles rule out

| row (E_def / realnoise) | STOI | PESQ | ΔSI-SDR | clears all three |
|---|---|---|---|---|
| **target** | ≥ 0.85 | ≥ 2.5 | ≥ 15 | |
| mask@24 bands | 0.915 / 0.949 | 1.908 / 2.152 | 11.66 / 12.62 | no |
| **mask@48 bands** ← today | 0.938 / 0.956 | 2.392 / 2.446 | 13.02 / 12.97 | no |
| mask@64 | 0.945 / 0.962 | 2.601 / 2.700 | 13.41 / 13.60 | no |
| mask@96 | 0.949 / 0.967 | 2.760 / 2.877 | 13.71 / 14.16 | no |
| mask@257 bins | 0.953 / 0.974 | 2.899 / 3.214 | 14.05 / 14.95 | **no** |
| mask@48 + phase < 1 kHz | 0.974 / 0.974 | 3.492 / 3.213 | 18.98 / 17.89 | yes |
| mask@96 + phase < 2 kHz | 0.987 / 0.986 | 4.102 / 3.834 | 21.98 / 20.89 | yes |
| **mask@96 + phase < 4 kHz** | 0.991 / 0.989 | 4.261 / 3.997 | 24.10 / 22.71 | yes |
| mask@257 + phase < 2 kHz | 0.991 / 0.993 | 4.352 / 4.328 | 23.22 / 22.85 | yes |

**Frequency resolution alone never reaches ΔSI-SDR 15.** A *perfect* 257-bin
mask reaches 14.05 / 14.95. Both sets agree. Resolution is not the lever.

**Phase is.** Substituting clean phase below 1 kHz alone — 32 of 257 bins —
takes ΔSI-SDR from 13.02 to 18.98 and PESQ from 2.392 to 3.492.

### What each survivor demands

G7-base on the identical clips: STOI 0.724, PESQ 1.278, ΔSI-SDR 9.29 — that is
**77.2% / 53.4% / 71.4%** of the ceiling it works against today. A plan is only
credible if it demands a similar fraction.

| plan | STOI need | PESQ need | ΔSI-SDR need | worst gap vs today |
|---|---|---|---|---|
| mask@48 + phase < 1 kHz | 87.3% | 71.6% | 79.0% | **+18.2 pts** |
| mask@48 + phase < 4 kHz | 86.5% | 64.8% | 68.3% | +11.4 pts |
| mask@96 + phase < 2 kHz | 86.1% | 60.9% | 68.2% | +8.9 pts |
| **mask@96 + phase < 4 kHz** | 85.8% | 58.7% | 62.2% | **+8.6 pts** |
| mask@257 + phase < 2 kHz | 85.7% | 57.4% | 64.6% | +8.5 pts |

Counter-intuitive and load-bearing: **the richer plans are easier to hit.** The
cheapest surviving architecture (phase below 1 kHz only) needs 71.6% of its PESQ
ceiling where we have never exceeded 53.4%. Buying more headroom lowers the
fraction you have to extract from it.

### What fits the board

`mac_by_bands.py`, forward-hook count, hop 256 at 62.5 fps:

| variant | params | MMAC/s (hook) | scaled to preflight accounting |
|---|---|---|---|
| 48 bands + fullband (today) | 49,663 | 20.9 | 32.3 (measured) |
| 96 bands | 44,060 | 36.4 | ≈ 56 |
| 256 bands | 44,060 | 96.1 | ≈ 149 |

Budgets: ESP32-S3 ≈ 200 MMAC/s per core; PS caps compute at **125 MMAC/s** and
model size at 200 KB. The hook counter reads 20.9 where the preflight measured
32.3 for the same configuration — it omits the STFT/ISTFT and counts transposed
convolutions by input rather than output — so the last column scales by that
1.55× ratio. **This is what eliminates the 257-bin plan: ≈149 MMAC/s breaks the
PS's own compute cap**, while 96 bands sits at ≈56.

Parameters do not move with the band count (the ERB matrix is fixed, the
convolutions are channel-wise), so every variant stays around 48 KB int8 against
a 200 KB limit. Compute is the only thing being spent.

### Chosen: 96 ERB bands + per-bin phase below 4 kHz

Highest ceiling that fits the PS compute cap, and the joint-lowest demand on
training (+8.6 pts worst case). ΔSI-SDR — the target that was impossible for
every magnitude-only design — arrives with **9.2 points of slack**: the plan
needs 62.2% of its ceiling where we already extract 71.4% of ours. The real work
is STOI (+8.6) and PESQ (+5.2).

**Why this does not repeat the phase branch's failure.** `gtcrn_lite.py:124`
records why phase was removed: the rotation was emitted per ERB band and applied
to every bin in it, "ill-posed in the wide ones" at bands up to **19 bins**. Two
things change here. The band count doubles — below 4 kHz at 96 bands the widest
band is **5 bins, mean 1.8** — and the new head is **per-bin**, not per-band, so
the ill-posedness is designed out rather than survived. That removal was also
scored on overall PESQ, the metric phase helps least; the +9.6 dB sits in
ΔSI-SDR at 0 dB, which was never the criterion it was judged on.

**Two build items this exposes, neither optional:** the full-band branch is
hardwired to the 48-band encoder width (`gtcrn_lite.py:235` asserts 6 vs 12 at
96 bands), so the band count is an architecture edit, not a flag; and a per-bin
phase head below 4 kHz has to be added and its MACs counted for real rather than
estimated.

**What this proves and what it does not.** Every row below a target is a proof
of impossibility — an oracle cannot be beaten by any model of its family, so no
training changes it. Every row above a target proves only that the target is not
ruled out. The demand table is the honest bridge between the two, and it is a
prior from one architecture's history, not a guarantee.

---

## Local confidence checks on the plan — and two corrections they forced

Four checks were run against the plan before handing it to anyone.
`E03/plan_confidence.py`, `E03/df_at_0db.py`, `E03/check_plan_arch.py`.

### Check 1 — bootstrap the margins. One claim did not survive.

10,000-resample percentile bootstrap over clips, 54 clips per set, seed 7:

| row | metric | mean | 95% CI | target | verdict |
|---|---|---|---|---|---|
| mask@257, E_def | ΔSI-SDR | 14.05 | [13.07, **15.18**] | 15 | **straddles** |
| mask@257, realnoise | ΔSI-SDR | 14.95 | [13.78, **16.19**] | 15 | **straddles** |
| mask@96 + phase <4 kHz, E_def | ΔSI-SDR | 24.10 | [23.12, 25.22] | 15 | clears |
| mask@96 + phase <4 kHz, realnoise | ΔSI-SDR | 22.71 | [21.66, 23.77] | 15 | clears |

**Correction.** The earlier statement that "resolution alone can never reach
ΔSI-SDR 15 — a perfect 257-bin mask reaches only 14.05" is **not supportable at
n = 54**. The interval crosses the target on both sets. The honest form is: a
perfect full-resolution mask lands *at* the target with the sampling error of
this measurement, so resolution alone is not a safe route — not that it is a
proven impossibility. Settling it needs more clips.

The chosen plan's margins are unaffected: every interval clears every target on
both sets, ΔSI-SDR by more than 6 dB at the lower bound.

### Check 2 — the deep-filter contradiction. My retirement of it was wrong.

`oracle_w24.json` and `oracle_ladder.json` disagree on the same row at the same
hold window (H = 3.297 vs 2.275). Re-measured with the current code at 0 dB
across the hold length the result is known to be sensitive to:

| row | STOI | PESQ | ΔSI-SDR | clears all three |
|---|---|---|---|---|
| B ideal@48, noisy phase | 0.938 | 2.392 | 13.02 | no |
| H deep filter <5 kHz + ERB, hold 8 | 0.986 | 3.996 | 22.57 | **yes** |
| H, hold 24 | 0.954 | 3.098 | 17.38 | **yes** |
| H, hold 48 | 0.939 | 2.849 | 16.11 | **yes** |

**Correction.** "This retires the deep-filter direction" was wrong, and it was
wrong because it read `oracle_ladder.json`, which was produced by the earlier
cascade the ladder's own docstring describes as "a different (and worse)
architecture" — the deep filter *replacing* the mask below 5 kHz rather than
refining it. The current cascade clears all three targets at every hold length
tested, and the source comment at `gtcrn_lite.py:193` was substantially right.

The deep filter is a **viable second lever**, not a dead one. It is still not
the chosen one, on demand rather than on ceiling: at the conservative hold 48 it
needs 90.5% / 87.8% / 93.1% of its own ceiling against the 77.2% / 53.4% / 71.4%
we currently extract — gaps of +13 to +34 points, where the chosen plan's worst
gap is +8.6. Note also that this oracle holds its coefficients fixed per clip,
so it understates what a per-frame predictor could do; the hold-8 row is the
fairer read of that family's real ceiling and is competitive.

### Check 3 — the architecture builds, fits, trains and stays causal

`check_plan_arch.py`, 11/11 passed. Both blockers have cheap fixes:

- **Full-band branch hardwired to 48 bands.** Fix: take the encoder's last
  stride from 1 to 2, so 96 → 48 → 24 → 12 → 6. The encoder width is 6 again,
  the full-band branch is untouched, the DPRNN sees exactly today's shape, and
  the decoder mirrors with strides (2,2,2,2). No resize, no padding, no new
  operator — every constraint that drove the original stride choices still holds.
- **No per-bin phase head.** The deep-filter head already has the geometry: a
  deconv chain from width 6 to 183 bins (5.7 kHz), per bin. 4 kHz is bin 128,
  inside it. The phase head is that chain with 2 output channels (cos, sin)
  instead of 2·taps — no atan2, which is the operator CMSIS-NN cannot run and
  one of the reasons the original branch was removed.

Measured, not estimated:

| | value | budget |
|---|---|---|
| parameters | **51,711** (base 49,663) | ~50 KB int8 vs 200 KB |
| compute | **48.0 MMAC/s** | PS cap 125; 24% of one ESP32-S3 core |
| gradients | every parameter finite and non-zero | |
| causality | rewriting frames 20+ moves frames 0–19 by **0.00e+00** | mask and phase head both |
| fresh head | max rotation **0.015 rad** | starts as identity |

The last one matters for the schedule: the head initialises as a no-op, so the
variant fine-tunes from G7-base rather than training from scratch.

### Where this leaves the plan

**Unchanged: 96 ERB bands + per-bin phase below 4 kHz.** It survived both
corrections — it was never the claim that failed — and it is now the only
candidate that is simultaneously (a) statistically clear of all three targets on
both sets, (b) the joint-lowest training demand, (c) measured at 48 MMAC/s and
51,711 parameters inside every budget, and (d) built and verified to train
causally from the current checkpoint.

---

## Pre-training gates — the phase plan does not survive them

`E03/gates_bc.py`, `E03/gate_c_headfix.py`, `E03/gate_c_deepfilter.py`
· `runs/g012/gates_bc.json`, `gate_c_headfix.json`, `gate_c_deepfilter_1500.json`

The oracle ladder says a *perfect* model of a family would pass. It says nothing
about whether this network, with this loss and this optimiser, can get there. The
argument used to pick the plan — "G7-base extracted 53% of its PESQ ceiling, so a
plan needing 59% is reachable" — is a heuristic borrowed from a different
architecture. These gates measure the actual model instead.

### Gate A — export · 2/2 PASS

| check | result |
|---|---|
| G7-base weights load into the 96-band variant | 108 tensors, 0 unmatched |
| ONNX export, opset 17 | works |

Weight *shapes* are compatible because the band count changes only the fixed ERB
buffer and the strides, never a weight tensor.

### Gate B — optimisation · 6/7 PASS

**B1 FAIL — the variant cannot be warm-started from G7-base.**

| on the 54 clips at 0 dB | STOI | PESQ | ΔSI-SDR |
|---|---|---|---|
| noisy input | 0.657 | 1.079 | 0.00 |
| G7-base | 0.724 | 1.278 | **+9.29** |
| 96-band variant, G7-base weights, untrained | 0.559 | 1.133 | **−1.42** |

Shape-compatible is not semantically compatible: the decoder was trained to emit
a mask on the 48-band grid, and on the 96-band grid the same weights are
meaningless — worse than passing the noisy input through. **Correction to the
plan as written: it needs a from-scratch run, not a 6-epoch fine-tune.** The
"starts as a near-copy of G7-base" claim was wrong.

**B2 PASS — the gradient is healthy and reaches the phase head.**

| branch | grad norm |
|---|---|
| erb/enc | 1.94e+01 |
| fullband | 1.34e+01 |
| fuse | 1.44e+01 |
| dprnn | 2.98e+01 |
| decoder | 3.28e+01 |
| **phase head** | **4.78e+00** |
| spp | 6.43e−05 |

No NaN or Inf. Phase/decoder ratio 0.146 — not vanishing. (The SPP head's
near-zero gradient is pre-existing, not introduced here.)

**B3 PASS — it overfits 16 clips easily.** Loss 5.675 → −9.911, SI-SDR on the
memorised set 0.48 → 12.42 dB over 600 steps. The architecture can learn.

### Gate C — learnability · FAIL, and the fix did not work

Everything frozen except the phase head; target is the per-bin rotation from
noisy phase to clean phase, magnitude-weighted — exactly what the oracle row
substitutes. Alignment 1.0 means the phase is recovered; the identity rotation
(leave the noisy phase alone) scores 0.8510.

| steps | 0 | 250 | 500 | 750 | 1000 | 1250 | 1499 |
|---|---|---|---|---|---|---|---|
| no skips | 0.8341 | 0.8517 | 0.8572 | 0.8620 | 0.8632 | 0.8654 | **0.8667** |
| + full-band skips | 0.7896 | 0.8511 | 0.8543 | 0.8607 | 0.8655 | 0.8646 | **0.8699** |

It beats the identity — by **+0.016**, after 1500 steps, on a 16-clip
*memorisation* task, and the increments are shrinking. The oracle prices perfect
phase at +9.6 dB of ΔSI-SDR; this recovers a small fraction of the way there.

The bottleneck hypothesis was that the head reads a 6-wide frequency bottleneck
and upsamples ~30×, while the decoder has skips and the head has none. Adding
skips from the full-band branch's own bin-domain activations gives **1.20×** —
not a fix. **The limit is not the bottleneck: the per-bin phase residual is not
predictable from this trunk.** That is consistent with why DeepFilterNet-family
work applies complex *filtering* instead of regressing phase directly.

### Gate C on the other lever — the deep filter PASSES

Identical protocol, identical budget: frozen trunk, 16 clips, 1500 steps, only
the 2,176-parameter deep-filter head trainable.

| steps | 0 | 250 | 500 | 750 | 1000 | 1250 | 1499 |
|---|---|---|---|---|---|---|---|
| SI-SDR (dB) | 3.50 | 9.29 | 9.85 | 10.20 | 10.38 | 10.51 | **10.60** |
| vs mask alone (9.14) | −5.64 | +0.15 | +0.72 | +1.06 | +1.24 | +1.37 | **+1.46** |

**+1.46 dB over the mask alone, still climbing at the cut**, from a head with
2,176 parameters and every other weight frozen. The phase head's comparable
result was +0.016 of alignment, decelerating.

### Consequence — the recommendation changes

**Drop the phase plan. Build the deep-filter cascade.** Not on ceiling — both
families clear the PS targets at oracle strength — but on *learnability*, which
is the thing the ceiling analysis could not see and the gates could:

- phase: ceiling 24.10 dB ΔSI-SDR, **reachable fraction measured as negligible**
- deep filter: ceiling 16.11–22.57 dB ΔSI-SDR depending on hold, **+1.46 dB
  demonstrated in 1500 steps on a frozen trunk with 2,176 parameters**

This is the third correction in this thread and it lands on the same conclusion
each time: **Suryansh's `df=True` was the right architectural call.** His run
failed on the training distribution (fixed 0 dB), not on the architecture — and
the full-range rerun already fixed most of that regression.

**Recommended next experiment, one variable:** G7-base + deep-filter head,
fine-tuned on the full-range −10..+20 dB dataset he has already built, evaluated
against our standard sets *and* stratified at 0 dB. That is his
`g7_finetune_fullrange` run with nothing changed except adding the 0 dB
stratified report — the checkpoint already exists.

**Still open, and it is the real risk:** at 0 dB his full-range checkpoint scored
STOI 0.804 / PESQ 1.582 / ΔSI-SDR ~8, against targets of 0.85 / 2.5 / 15. Gate C
shows the deep filter *can* climb; it does not show it climbs far enough. Nothing
short of the training run answers that.

---

## The full-range deep-filter checkpoint, measured at 0 dB

`E03/model_at_0db.py` on `runs/g7_finetune_fullrange/best.pt`, fetched from
`SuryanshSinha2801/rhear-g7-pr@ae5a549`. 54 clips at 0 dB ±2.5 dB, both sets.
`runs/g012/model_at_0db_ft_edef.json`, `model_at_0db_ft_realnoise.json`.

| checkpoint | set | STOI | PESQ | ΔSI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| G7-base | E_def | 0.724 | 1.278 | 9.29 | 13.79 | 8.18 |
| G7-base | realnoise | 0.820 | 1.301 | 9.18 | 14.42 | 8.76 |
| full-range, df=True | E_def | 0.710 | 1.241 | 8.91 | 14.35 | 7.67 |
| full-range, df=True | realnoise | 0.816 | 1.287 | 9.11 | 15.27 | 8.48 |
| **PS target** | | **0.85** | **2.5** | **15.0** | | |

Delta vs G7-base at 0 dB:

| set | STOI | PESQ | ΔSI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def | −0.014 | −0.037 | −0.39 | **+0.57** | **−0.51** |
| realnoise | −0.004 | −0.014 | −0.07 | **+0.86** | **−0.28** |

The same SI-SIR-up / SI-SAR-down signature as the 0 dB run, weaker again but not
gone. **The deep-filter checkpoint is still marginally behind G7-base at the
condition the PS grades.** G7-base remains current best.

### Distance to target, as a fraction of the deep filter's own ceiling

Against H at hold 24 (STOI 0.954, PESQ 3.098, ΔSI-SDR 17.38):

| | STOI | PESQ | ΔSI-SDR |
|---|---|---|---|
| achieved, E_def | 74.5% | 40.1% | 51.2% |
| achieved, realnoise | 85.5% | 41.5% | 52.4% |
| **needed** | **89.1%** | **80.7%** | **86.3%** |

STOI is within reach — realnoise is at 85.5% of 89.1%. PESQ and ΔSI-SDR are at
roughly half of what the targets demand.

### The finding that matters, and it is about training, not architecture

Two measurements of the *same* deep-filter head, on the same trunk:

| | what trains | result |
|---|---|---|
| Gate C | **head only**, 2,176 params, trunk frozen | **+1.46 dB** over mask alone, still climbing |
| this checkpoint | everything, jointly, 24 epochs | **−0.39 / −0.07 dB** vs no head at all |

When only the head trains, it earns +1.46 dB. When the whole network trains
jointly, the head earns nothing. The head is not the problem — it demonstrably
works when it is the only thing that can improve the loss.

**Hypothesis, consistent with both numbers but not yet proven:** joint training
lets the mask absorb the capacity, and nothing ever drives the deep-filter head
to do the job the oracle says it can do. The residual parameterisation makes this
easy to fall into — the head starts at identity, so leaving it at identity is
always a valid local solution.

**The test is cheap and it is the next experiment:** staged training. Train the
mask to convergence, freeze it, train the deep-filter head alone the way Gate C
did, then unfreeze both at a low learning rate. If the +1.46 dB survives into a
jointly-trained checkpoint, the deep filter is the lever. If it evaporates the
moment the trunk unfreezes, that is worth knowing before more epochs are spent.

**Stated plainly: no checkpoint we have reaches any PS target at 0 dB.** The
closest is STOI on realnoise (0.816 against 0.850). PESQ (1.287 against 2.500)
and ΔSI-SDR (9.11 against 15.00) are not close, and nothing measured so far
shows a path that closes them inside this project's remaining time.

---

## Is it us, or the test set? Three published models answer it

`E03/crossbench.py`, `E03/crossbench_metricgan.py`
· `runs/g012/crossbench/*.json`

Published models, their own pretrained weights, their own published inference
paths, run over OUR held-out defence clips and scored by OUR scorer:

| model | params | published PESQ on VoiceBank+DEMAND | our E_def, all 300 | | our E_def at 0 dB | |
|---|---|---|---|---|---|---|
| | | | **PESQ** | **STOI** | **PESQ** | **ΔSI-SDR** |
| GTCRN, VCTK ckpt | 23.7k | 2.87 | 1.444 | 0.765 | — | — |
| GTCRN, DNS3 ckpt | 23.7k | 2.87 | 1.643 | 0.818 | — | — |
| MetricGAN+ | ~1.9M | 3.15 | 1.820 | 0.755 | 1.442 | **−0.13** |
| **G7-base (ours)** | **49.7k** | — | **1.635** | **0.815** | **1.278** | **+9.29** |
| **PS target at 0 dB** | | | | | **2.50** | **+15.00** |

**Nothing is wrong with our model.** Three published systems — one of them 38×
larger and a GAN trained to maximise PESQ directly — all land in the 1.44–1.82
band on our data. None comes close to 2.5. MetricGAN+ scores higher PESQ than us
because it optimises PESQ directly, and pays for it everywhere else: STOI 0.755
vs our 0.815, and ΔSI-SDR of **−2.97 dB overall / −0.13 dB at 0 dB** against our
+9.29 — it makes the signal worse in SI-SDR terms while making it sound better
to PESQ.

### Why the benchmark numbers do not transfer

VoiceBank+DEMAND's test set — the source of nearly every headline PESQ figure in
the field — is mixed at **2.5 / 7.5 / 12.5 / 17.5 dB**, a mean near 10 dB, over
five benign noises (bus, cafe, office, public square, living room), on clean
studio speech, with no clipping, no preamp distortion, no microphone mismatch.

Our evaluation is at **0 dB**, on MAD/DEMAND defence noise, with 15% clipping,
10% preamp distortion, microphone tilt, level mismatch and mic self-noise.

Published results that *do* report at 0 dB on ordinary noise land around
**PESQ 1.83–2.39**. We have found no published system reporting PESQ ≥ 2.5 at
0 dB on impulsive or military noise.

**So the PS's "PESQ ≥ 2.5" is very likely quoted from benchmark-average
conditions, not from 0 dB.** Our own numbers agree: G7-base scores PESQ 2.44 in
the > 15 dB bucket of E_def — benchmark-like conditions, benchmark-like result.

### The one lever this measurement does identify

The same GTCRN architecture scores **1.444 with VCTK-DEMAND training and 1.643
with DNS3 training** — +0.199 PESQ and +0.053 STOI from nothing but a more
diverse training corpus. That is the largest single improvement attributable to
one change anywhere in this project's log, and it is a **data** lever, not an
architecture one.

### What to report

Report both conditions and the cross-benchmark table. "We match published
state-of-the-art on our own data, and here is what those same systems score on
it" is a defensible and unusual claim. Reporting a mix-average PESQ against a
target defined at 0 dB is not, in either direction.

---

## Evaluated against PS 26052 as the PS words it

`E03/eval_ps.py` · `runs/g012/eval_ps.json`

The PS describes the **dataset** as covering "varying SNR levels" and then states
three targets flatly — `SNR > 15 dB`, `STOI > 0.85`, `PESQ > 2.5`. It names no
evaluation SNR and no test set. So the primary number is the **whole mixed-SNR
test set**, not a slice: our sets span −10..+20 dB with a mean of +4.7 dB, which
is exactly the condition described. Earlier entries in this document graded at
0 dB; that was our own stricter reading, and it is not what the PS asks for.

| set | cut | n | STOI | PESQ | out SNR | ΔSNR | met |
|---|---|---|---|---|---|---|---|
| E_def | all | 300 | 0.815 | 1.635 | 10.05 | 7.54 | 0/3 |
| E_def | no clipping/distortion | 241 | 0.819 | 1.651 | 10.34 | 7.30 | 0/3 |
| E_def | SNR ≥ 2.5 dB | 177 | **0.911** | 1.925 | 14.37 | 6.17 | 1/3 |
| realnoise | all | 300 | 0.843 | 1.613 | 9.61 | 7.80 | 0/3 |
| realnoise | no clipping/distortion | 229 | **0.850** | 1.634 | 10.03 | 7.87 | 1/3 |
| realnoise | SNR ≥ 2.5 dB | 160 | **0.933** | 1.958 | **15.10** | 6.65 | 2/3 |

**PESQ is the only real gap.** STOI reaches 0.933 and output SNR reaches 15.10 dB
at benign SNR — both targets met on realnoise at the SNR range published PESQ
figures are measured over. PESQ never gets above 1.96 anywhere.

**A hypothesis this kills:** removing our own clipping and preamp-distortion
augmentation from the test set buys almost nothing — **+0.004 STOI, +0.016
PESQ**. The harshness of our evaluation is the *noise and the SNR*, not the
capture degradations. Worth knowing, because it was the obvious suspect.

### What the PESQ gap would take

Over the full mix the oracle ceiling for our architecture is **PESQ 2.838**
(ideal 48-band mask, `oracle_ladder.json`), so the 2.5 target *is* inside this
model class at the PS's stated condition — unlike at 0 dB, where the ceiling is
2.392 and the target is unreachable. We sit at 1.635, which is **57.6% of that
ceiling**; the target needs **88.1%**.

Levers, ranked by measured evidence rather than by expectation:

| lever | evidence | verdict |
|---|---|---|
| training-data diversity | GTCRN, same architecture, VCTK→DNS3 weights: **+0.199 PESQ, +0.053 STOI** on our data | largest single-change gain measured anywhere in this project |
| frequency resolution 48→96 bands | ceiling 2.392→2.760 at 0 dB; params unchanged, 36 MMAC/s | raises the ceiling, cheap |
| deep-filter head | +1.46 dB SI-SDR on a frozen trunk in 1500 steps (Gate C) | works when trained alone; absorbed by the mask in joint training |
| perceptual loss | **spent.** G1 (`perc 1.0`) 1.493 vs G0 1.604; G5 (`perc 0.15`) 1.617 vs G2 1.629 | too strong hurts, mild is a wash |

**Nothing here closes a 0.9 PESQ gap before 7 Sep**, and the honest position for
the round is that STOI and output-SNR are met at benign SNR while PESQ is short,
with the ceiling analysis showing exactly how much of the shortfall is
architectural.

---

## VoiceBank+DEMAND fine-tune — two of three targets cleared, PESQ lands *at* the line

`E03/prep_vbdemand_parquet.py`, `E03/finetune_vbdemand.py`, `E03/eval_vbdemand.py`,
`E03/vbd_margin.py` · `runs/g7_vbdemand/best.pt`

G7-base, architecture unchanged (49,663 params, hop 256), fine-tuned on
VoiceBank+DEMAND's own 28-speaker training set — 10,700 pairs, 26 speakers, with
**p286 and p287 held out entirely for checkpoint selection**. 12 epochs, best
validation SI-SDR 14.31 dB. The 824-utterance test set was scored **once**.

Corpus came from `JacobLinCool/VoiceBank-DEMAND-16k` on HuggingFace: the
Edinburgh datashare host stopped serving mid-download, going from working to
returning zero bytes and leaving both training zips truncated at ~10%. The HF
copy is already at 16 kHz, which removed the resampling step.

| | noisy in | enhanced | 95% CI | target | verdict |
|---|---|---|---|---|---|
| STOI | 0.921 | **0.936** | [0.932, 0.940] | > 0.85 | **clears** |
| PESQ (wb) | 1.967 | **2.524** | [2.481, 2.567] | > 2.5 | **straddles** |
| output SI-SDR | 8.45 | **18.35** | [18.06, 18.63] | > 15 | **clears** |

10,000-resample bootstrap over clips, seed 11.

**The honest claim is two of three cleared, with PESQ at the threshold.** The
point estimate is over the line and the fine-tune bought a real +0.170 PESQ over
zero-shot, but the interval contains 2.5 — 824 utterances cannot distinguish
2.524 from exactly 2.5. Saying "meets all three" would be the same overclaim the
bootstrap caught twice before in this document.

Under the improvement reading of "SNR > 15 dB", ΔSI-SDR is **+9.90 dB** and does
not meet it. Report both readings.

### The domain trade, measured

The same fine-tuned checkpoint, on our defence sets:

| set | cut | STOI | PESQ | out SNR |
|---|---|---|---|---|
| E_def | all | 0.789 | 1.495 | 6.91 |
| realnoise | all | 0.814 | 1.513 | 6.26 |
| | *(G7-base, same cut)* | *0.815 / 0.843* | *1.635 / 1.613* | *10.05 / 9.61* |

Fine-tuning on the benchmark costs **−0.14 PESQ and −3.1 dB output SNR** on
defence noise. **There are now two checkpoints for two domains and neither is
universally better** — `g7_hop256_50k` is the defence model,
`g7_vbdemand` is the benchmark model. Presenting the benchmark number as if it
described defence performance would be dishonest, and the two-row table above is
the antidote.

### Continued to 20 epochs — the plateau is real

`runs/g7_vbdemand_x24/best.pt` · `runs/g012/vbd_margin_x24.json`

Validation was still creeping at epoch 12 (14.25 → 14.30 → 14.31), so training
continued at lr 2e-4 with selection still on the held-out speakers. It
**early-stopped at epoch 8** of the continuation — four epochs without
improvement — after the scheduler had cut the rate three times to 2.5e-5. Best
validation SI-SDR moved 14.31 → 14.34.

Both test scorings, reported together rather than picking the better one:

| checkpoint | STOI | PESQ | 95% CI on PESQ | output SI-SDR |
|---|---|---|---|---|
| zero-shot (no fine-tune) | 0.931 | 2.354 | — | 17.71 |
| 12 epochs | 0.936 | 2.524 | [2.481, 2.567] | 18.35 |
| + 8 more epochs | 0.936 | **2.529** | [2.487, 2.570] | 18.34 |

**Eight further epochs bought +0.005 PESQ and the interval still contains 2.5.**
STOI and output SI-SDR are unchanged to three figures. This axis is exhausted:
more training on VoiceBank+DEMAND does not move the metric that matters, and the
conclusion is stable across two independently selected checkpoints.

**Final position on the benchmark: STOI and output-SNR cleared with confidence;
PESQ reaches the target and cannot be shown to exceed it.** Raising it further
needs a higher ceiling — frequency resolution or training-data diversity — not
more epochs.
