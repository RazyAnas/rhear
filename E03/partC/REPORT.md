# Part C + E report — quiet-speech forensics and inference-time ablations

**No retraining. Frozen baseline verified unchanged** (`model.pt` sha `1c1452e4bdc5c848`,
eval manifest sha `54fa4551ca3dcf01`). Demo package untouched.

**PROVENANCE:** noise in this evaluation set is **100% synthetically generated**
(5,211/5,211 layers). Speech is real LibriSpeech. Speech-path findings are
expected to transfer. **Siren-specific conclusions are deferred** — our "siren"
is a single swept sinusoid with no harmonics, AM or reverb. Real-noise
evaluation takes priority once MUSAN lands.

---

## 1. Does quiet-speech suppression definitely occur?

**Yes — systematic, and the mechanism is SNR-dependent, not level-dependent.**

Gain the model actually applies, by input SNR (300 clips, fixed split):

| input SNR | speech gain | noise gain | discrimination | ΔSTOI |
|---|---|---|---|---|
| −10…−5 dB | **−7.74 dB** | −11.50 dB | **3.76 dB** | +0.026 |
| −5…0 | −5.76 | −11.67 | 5.91 | +0.036 |
| 0…5 | −4.82 | −10.71 | 5.89 | +0.046 |
| 5…10 | −3.87 | −10.66 | 6.79 | +0.028 |
| 10…15 | −2.25 | −10.72 | 8.47 | +0.013 |
| 15…20 | **−1.31** | −10.28 | **8.97** | +0.002 |

Speech is attenuated **6.4 dB more at −7.5 dB SNR than at +17.5 dB SNR**
(correlation of speech gain with SNR **+0.678**). Noise attenuation is nearly
flat (−10.3 to −11.7 dB), so **the model does not become more aggressive at low
SNR — it becomes less discriminating**: 8.97 dB of speech/noise separation at
high SNR collapses to 3.76 dB at low SNR.

## 2. Where in frequency and time?

**Across clips by SNR, not within a clip by level.** The within-clip retention
curve is shallow — 0.86 at −28 dB below the clip's speech peak vs 0.99 at the
peak. A ~13% relative loss, not a collapse.

Two findings that **contradict the usual assumptions**:

| | gain | |
|---|---|---|
| vowel-like frames | −4.17 dB | |
| consonant-like frames | **−3.84 dB** | consonants preserved *better*, not worse |
| below 1 kHz | **−4.75 dB** | |
| above 1 kHz | −3.99 dB | low frequencies attenuated *more* |

So this is **not** the classic "consonants and high frequencies get eaten"
failure. It is broadband, mildly LF-weighted, and driven by SNR.

**18% of clips (54/300) get WORSE STOI**, and those clips have a *higher* mean
SNR (+5.7 dB) than the ones that improve (+4.4 dB) — the model damages speech
that was already fairly clean.

## 3. Which inference-time ablation helps most?

Full 300-clip fixed set. Noisy reference: STOI 0.805, PESQ 1.382, SI-SDR +2.61.

| variant | STOI | PESQ | SI-SDR | speech | noise | ΔSTOI | ΔPESQ |
|---|---|---|---|---|---|---|---|
| A0 baseline | 0.830 | 1.621 | 8.53 | −4.14 | −10.96 | — | — |
| A1 noisy phase | 0.834 | 1.655 | **8.95** | −4.14 | −10.96 | +0.004 | +0.034 |
| **A1+A2 α=0.8** | **0.837** | **1.659** | 8.86 | −3.62 | −9.97 | **+0.008** | **+0.038** |
| A1+A2 α=0.7 | 0.839 | 1.654 | 8.75 | −3.32 | −9.40 | +0.009 | +0.033 |
| A1+A2 α=0.6 | 0.839 | 1.624 | 8.58 | −2.99 | −8.74 | +0.010 | +0.003 |
| A1+A2 α=0.5 | 0.839 | 1.593 | 8.31 | −2.62 | −7.97 | +0.010 | −0.028 |

**Recommended: A1 (noisy phase) + A2 (mask exponent α=0.8).** STOI saturates at
0.839 for α ≤ 0.7, so the extra suppression loss below α=0.8 buys nothing —
α=0.8 has the best PESQ and gives up the least noise attenuation.

**A1 alone is a free win**: it improves all three metrics, costs zero compute,
and removes a head from the inference path.

## 4. What does NOT help

**Every gain-floor variant hurt STOI.** This is a clean negative result and it
contradicts the intuition behind a "minimum gain for speech-likely regions":

| variant | ΔSTOI |
|---|---|
| A3 floor 0.1 / 0.2 / 0.3 | −0.007 / −0.013 / −0.014 |
| A4 speech-band floor 0.2 / 0.3 | −0.013 / −0.014 |
| A6 SPP-conditioned floor 0.2 / 0.3 | −0.007 / −0.008 |
| A7 low-SNR floor 0.25 / 0.4 | −0.013 / −0.021 |
| A5 temporal smoothing 3 / 5 frames | −0.002 / −0.005 |

**Why floors fail and the exponent works.** The measured mask distribution is
strongly bimodal — p25 = 0.003, p50 = 0.131, p75 = 0.621. A floor lifts
*confidently-zeroed* bins uniformly (0.001 → 0.2), readmitting noise the model
had correctly removed. The exponent scales proportionally and preserves those
zeros (0.001^0.8 = 0.004) while lifting the uncertain mid-range. The floors were
tested at values taken from that measured distribution, not invented.

## 5. Does the best ablation improve STOI without unacceptable noise loss?

**Yes.** A1+A2 α=0.8 versus baseline:

- STOI **+0.008**, PESQ **+0.038**, SI-SDR **−0.09** (8.86 vs 8.95 for A1 alone)
- speech retained **0.52 dB better** (−3.62 vs −4.14 dB)
- noise attenuation costs **1.0 dB** (−9.97 vs −10.96 dB)

Half a dB more speech for one dB less noise suppression, with PESQ up. For a
product whose stated priority is intelligibility over dB, that is the right side
of the trade.

## 6. Retraining

**None performed.** No architecture, dataset, loss, frame size, ERB band count,
phase handling or optimiser was changed. All results are inference-time only, on
the frozen checkpoint, on the fixed 300-clip speaker-disjoint split.

---

## What this does and does not license

The dominant mechanism is **loss of speech/noise discrimination at low SNR**
(8.97 → 3.76 dB), not excessive suppression per se — the model's noise gain is
essentially constant across SNR. That points at a *training* problem
(discrimination) rather than a *decision-rule* problem (suppression), so an
inference-time guard can only partly compensate. It does compensate: +0.008
STOI, +0.038 PESQ for 1 dB of noise attenuation.

**Not yet established:** anything about real sirens, real rotor noise or real
defence-noise generalisation. Those require the real corpus.
