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
