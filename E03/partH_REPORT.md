# Part H — retrain on real noise. ONE variable changed.

**Changed:** training-noise domain, synthetic → real.
**Held identical:** architecture, loss (asymmetric ρ=8), frame size, ERB bands,
optimiser, LR schedule, epochs, batch size, dataset size (2000/300), speaker
split, SNR range, checkpoint-selection rule.

Training data: 603 real MUSAN point-source recordings + 126 real measured RIRs.
**Leakage verified 0** on both noise sources and RIR rooms against the real test
set (asserted in the build, not merely checked).

## Result — evaluated on the frozen 300-clip real-noise test set

| | frozen (synthetic-trained) | **real-trained** | gain |
|---|---|---|---|
| Δ STOI | +0.000 | **+0.019** | **+0.019** |
| Δ PESQ | +0.085 | **+0.186** | **+0.100** |
| Δ SI-SDR | +2.24 dB | **+6.32 dB** | **+4.09 dB** |

**And it did not cost anything on the synthetic set** — this is a genuine
improvement, not a domain trade:

| | frozen | real-trained | change |
|---|---|---|---|
| Δ STOI | +0.024 | +0.021 | −0.004 |
| Δ PESQ | +0.239 | **+0.277** | +0.038 |
| Δ SI-SDR | +5.92 dB | **+7.26 dB** | +1.34 dB |

Training on real noise improves PESQ and SI-SDR on *both* domains and costs
0.004 STOI on synthetic — within noise. The domain-mismatch hypothesis is
confirmed and the fix generalises in both directions.

## Prediction check

Recorded before the run: *"real ΔSTOI should land in +0.010…+0.025."*
**Actual: +0.019.** The prediction held.

## A limitation of this run that must not be glossed over

**27.59 % of training batches (1890 of 6850) were skipped as non-finite.**

Diagnosed: it is **not** the near-silent-target guard — 0.0 % of clean targets in
either dataset fall below that threshold. The skips come from non-finite loss or
gradient norms, i.e. **numerical instability in training**, not bad data.

So this result was achieved while discarding roughly a quarter of the available
optimisation steps. Fixing the instability is the obvious next improvement and
is expected to help further. **The model is not "properly trained" yet, and the
numbers above should be read as a floor, not a ceiling.**

## Still below the PS targets

| | achieved (real noise) | PS target |
|---|---|---|
| STOI | 0.818 | > 0.85 |
| PESQ | 1.481 | > 2.5 |
| SI-SDR | +8.12 dB output | > 15 dB |

Context: the measured oracle ceiling for this architecture is STOI **0.967**, so
the design is not the limit. Remaining gap is training quality (the 27.6 % skip
rate), data volume (2.9 h), and noise-class diversity — this training set is
100 % `environmental`, with no engine, rotor, siren or military classes. MUSAN
proper and MAD address the last of those.

## Provenance

Speech: real LibriSpeech. Noise: **real recordings** (MUSAN point-source).
Rooms: **real measured RIRs**. This is the first RHEAR result trained and
evaluated entirely on real recorded audio. It still licenses **no claim** about
real sirens, rotor or military noise — those classes are not in this corpus.

---

# Part H.2 — numerical-stability fix

**Cause (measured, not guessed).** Not `angle()`, not bad data — 0.0 % of clean
targets fall below the near-silent guard, and a 62-step reproduction of epoch 1
produced **zero** skips. The cause is the power-law compression's own gradient:
d/dS of `|S|^0.3` goes as `|S|^-0.7`, which diverges as a bin approaches zero.

That explains the timing. Early in training the mask sits near 0.5 everywhere and
nothing is near zero. Late in training it becomes bimodal (measured p25 = 0.003),
many bins reach zero, and the gradient explodes. **The failure was a consequence
of the model learning to suppress confidently.**

**Fix — two numerical changes, objective unchanged where it matters:**

| | before | after |
|---|---|---|
| epsilon in the compression | 1e-8 | **1e-4** |
| max &#124;gradient&#124; | 259.7 | **10.1** |
| loss change above −60 dBFS | — | 3.4e-3 |

plus replacing `(|S|+eps)^c · exp(j·angle(S))` with the algebraically identical
`S · (|S|+eps)^(c-1)` (max difference 2.6e-7), which drops `angle()` — worth a
further 3.5× near zero.

**Result: skip rate 27.59 % → 0.00 %.** The instability is gone, verified with
per-epoch logging. Training also ran faster (129 min vs 171 min).

## But it was not the bottleneck

| Δ vs noisy, real-noise test set | frozen | real-trained | **+ stabilised** | gain |
|---|---|---|---|---|
| STOI | +0.000 | +0.019 | **+0.022** | +0.003 |
| PESQ | +0.085 | +0.186 | **+0.198** | +0.012 |
| SI-SDR | +2.24 | +6.32 | **+6.57 dB** | +0.25 |

**Prediction check.** Recorded before the run: *"if the 27.6 % of discarded steps
were costing real progress, ΔSTOI should improve beyond +0.019; if it lands at
roughly +0.019, the skipped batches were not the bottleneck."*

**Actual +0.022 — a +0.003 gain.** The prediction resolves toward the second
branch: **the discarded batches were not what was holding the model back.** The
fix was correct and necessary — a 27.6 % skip rate is not something to ship — but
it bought little.

## What that leaves

The remaining gap to the PS targets (STOI 0.821 vs 0.85) is therefore **not**
training stability, and **not** the architecture (measured oracle ceiling 0.967).
By elimination the candidates are **data volume** (2.9 h), **noise-class
diversity** (this corpus is 100 % `environmental` — no engine, rotor, siren or
military classes), and **model capacity** (23 k parameters). Those have not been
tested and should not be asserted as the cause until they are.
