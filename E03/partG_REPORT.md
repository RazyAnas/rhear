# Part G (early) — real-noise evaluation of the frozen checkpoint

Run ahead of schedule because RIRS_NOISES completed and ships **843 real MUSAN
point-source noise recordings (5.91 h, 16 kHz)** and **417 real measured RIRs**.
MUSAN proper is still downloading (5.6/10.3 GB).

**No retraining. Frozen checkpoint unchanged.**

| | evaluation set |
|---|---|
| speech | REAL LibriSpeech, held-out test speakers |
| noise | **REAL recordings**, 85 distinct sources, 100% `real recording` |
| rooms | **REAL measured RIRs** on 178/300 clips |
| leakage | none possible on the noise side — the model trained on 100% synthetic noise, so every source here is unseen |

## 1. The headline: the model does not transfer

Same frozen checkpoint, synthetic vs real noise:

| | synthetic | **real** | drop |
|---|---|---|---|
| Δ STOI | +0.024 | **+0.000** | −0.024 |
| Δ PESQ | +0.239 | **+0.085** | −0.153 (−64 %) |
| Δ SI-SDR | +5.92 dB | **+2.24 dB** | −3.68 (−62 %) |

**The entire STOI benefit disappears on real noise.** PESQ and SI-SDR keep about
a third of their gain. This is a pure train/test domain mismatch — the model
trained on synthesised noise only.

## 2. Why: the mask collapses into uncertainty

Predicted mask distribution, same model, two domains:

| percentile | p10 | p25 | p50 | p75 |
|---|---|---|---|---|
| synthetic noise | 0.001 | 0.003 | 0.131 | 0.621 |
| **real noise** | 0.019 | **0.171** | **0.406** | 0.700 |

On synthetic noise the mask is strongly **bimodal** — the model confidently zeroes
half the bins. On real noise it sits in the **middle**: it does not know what to
remove. Consequently noise attenuation falls from **−10.96 dB to −7.43 dB**.

## 3. Does the Part C mechanism survive? Yes

| | synthetic | real |
|---|---|---|
| corr(speech gain, SNR) | +0.678 | **+0.565** |
| discrimination, low SNR | 4.79 dB | **2.41 dB** |
| discrimination, high SNR | 8.67 dB | 5.82 dB |

The SNR-dependent discrimination failure diagnosed in Part C **holds on real
noise** and is roughly twice as severe. That was worth checking rather than
assuming.

**But the failure changes character.** On real noise the within-clip retention
curve is **≥ 1.0 at every level** (1.56 at the quietest bins). Retention above 1
means the output exceeds the clean signal — **noise leaking into quiet speech
bins**, not speech being suppressed. On real noise the problem is under-
suppression, not over-suppression.

## 4. Do the Part E conclusions survive? Partly, and the useful one does

| variant | Δ STOI synthetic | Δ STOI real |
|---|---|---|
| **A1 drop predicted phase** | **+0.004** | **+0.005** |
| A2 mask exponent | +0.006…+0.010 | +0.003…+0.005 |
| A3/A4/A6/A7 gain floors | −0.007…−0.021 | −0.000…−0.004 |
| A5 temporal smoothing | −0.002…−0.005 | −0.000…−0.002 |

**A1 is robust across both domains** — it improves STOI, PESQ and SI-SDR in each,
costs nothing, and removes a head from the inference path. That is a real,
transferable result.

**The gain-floor negative result is now confirmed in two independent domains.**
No floor variant helped anywhere.

The A2 exponent still helps STOI on real noise but costs more PESQ there, so the
α=0.8 operating point chosen on synthetic data should be re-tuned on real data
rather than carried over.

## 5. What this changes

The dominant limitation is **the training domain, not the architecture and not
the decision rule.** Evidence:

- the oracle ceiling for this architecture is STOI **0.967** (measured earlier);
- on real noise the trained model achieves **+0.000**;
- the gap is not recoverable by any inference-time rule we tested — the best is
  +0.005.

So the next experiment is **retrain on real noise**, not tune a guard. Any
further diagnosis on synthetic noise would be measuring the wrong thing, exactly
as the sequencing correction predicted.

## 6. Still not established

Nothing here licenses claims about **real sirens, real rotor, real vehicle or
real military noise**. The real noise used is MUSAN free-sound environmental
recordings. MAD (military) and the full MUSAN taxonomy are still pending.
