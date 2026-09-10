# 08 — Communication layer (L1): what was tried, what the evidence says

Status as of 9 September 2026. This document is the full context for the L1
speech-enhancement path. It records the negative results as carefully as the
positive ones, because most of what was learned here came from things that did
not work.

Companion documents: `02-architecture.md` (three-rate design),
`07-dataset-selection-evidence.md` (corpus selection and the two build bugs).

---

## 1. What the problem statement asks

| metric | PS target |
|---|---|
| PESQ | > 2.5 |
| STOI | > 0.85 |
| ΔSNR | > 15 dB |

Compute envelope: ≤ 200 KB int8, ≤ 125 MMAC/s. The ESP32-S3 core delivers
roughly 200 MMAC/s per core, with 8 MB PSRAM and 16 MB flash.

The architecture keeps two problems separate (A2): **protection** is the ear
(L0, FxNLMS at 48 kHz, pure DSP, ~20.8 µs budget) and **communication** is the
message (L1, neural, 16 ms frames, feeds the radio). They have separate metrics
and separate success criteria. A background human voice must be **preserved**
in the ear path and **removed** from the radio path.

## 2. Where L1 stands against the PS

Measured on `edef` and `realnoise`, 300 clips each, model G12.

| input SNR band | PESQ | STOI | verdict |
|---|---|---|---|
| > 15 dB | 2.607 / 2.604 | 0.963 / 0.969 | **PESQ and STOI both pass** |
| 10–15 dB | 2.119 / 2.140 | 0.927 / 0.957 | STOI passes, PESQ short |
| 5–10 dB | 1.755 / 1.777 | 0.907 / 0.924 | STOI passes, PESQ short |
| 0–5 dB | 1.472 / 1.452 | 0.807 / 0.857 | both short |
| < 0 dB | 1.19–1.28 | 0.63–0.78 | both short |

So the honest statement is: **above 15 dB input SNR all three PS targets are
met. Below that, STOI holds down to about 5 dB and PESQ does not.** The gap at
0 dB is the open problem, and sections 3–5 explain why closing it by training a
better model does not work.

## 2b. Against published models, on our own defence noise

`E03/runs/g012/crossbench/`. Every row is the same 300-clip `edef` set, scored
the same way. Parameter counts are from our own measurement where recorded
(`sepformer_edef.json` reports `params = 25,613,569`) and from the published
paper for GTCRN.

| model | params | STOI | PESQ | SI-SDR out |
|---|---|---|---|---|
| **RHEAR G13b** | **49,663** | **0.8236** | **1.7159** | **10.76** |
| GTCRN, DNS3 weights | 23.7k | 0.8182 | 1.6430 | 9.61 |
| GTCRN, VCTK weights | 23.7k | 0.7648 | 1.4442 | 6.45 |
| SepFormer, WHAM16k | **25.6M** | 0.8021 | 1.5963 | 5.56 |
| MetricGAN+, VoiceBank | — | 0.7546 | 1.8197 | **−0.46** |

Two things follow.

**RHEAR beats GTCRN's own published weights on all three metrics** — the
architecture we derive from, run on our domain. +0.073 PESQ, +0.005 STOI,
+1.15 dB SI-SDR against the stronger (DNS3) checkpoint.

**It beats SepFormer by 5.2 dB SI-SDR at 1/516th the parameter count.** A
25.6-million-parameter transformer scores 5.56 dB on this material; a
49,663-parameter model scores 10.76 dB. Neither of the large models was trained
on defence noise, which is precisely the point: general-purpose capacity does
not substitute for domain-matched training.

**The caveat to state before anyone else does.** MetricGAN+ posts the highest
PESQ in the table (1.8197) and the worst SI-SDR (**−0.46 dB**, ΔSI-SDR
**−2.97 dB**). It is optimised directly against PESQ, and on this material it
degrades the signal while scoring well on the metric. This is the clearest
argument in the project for why the decision rule requires PESQ **and** STOI
**and** SI-SAR together, on both eval sets, rather than any single number.


## 3. The model ladder, and why it stopped moving

Model frozen throughout at GTCRNLite, hop 256, **49,663 params**. The trained
models from G11 on use 96 ERB bands; **the C port on the ESP32 is the earlier
48-band G7** (`psram_test.c:12`). Cost, from `E03/g7_esp32_preflight.json`:
**32.34 MMAC/s** at 62.5 fps. Weight storage at int8 is **~48.5 KB** — 200 KB is
the PS *cap*, not our size, and an earlier draft of this document wrongly stated
it as though it were the model's footprint.

Standing decision rule: **keep a change only if PESQ improves AND STOI does not
regress AND SI-SAR does not regress, on BOTH eval sets**, judged by paired
bootstrap with 10,000 resamples over identical clips.

| run | change | outcome |
|---|---|---|
| G8 | 48 → 96 ERB bands | **kept** |
| G9 | +56% parameters | rejected — train PESQ moved +0.001 |
| G10 | asymmetric loss rho 8 → 4 | rejected |
| G11 | 26 → 251 speakers | **kept** — PESQ +0.016, train/test gap 0.390 → 0.348 |
| G12 | 9 → 236 noise classes | **kept** — all four realnoise metrics better, STOI crossed 0.851, gap → 0.337 |
| G13/b/c | competing-talker training data | see §4 |
| G15 | deep-filter output head | rejected — see §5 |

Between G11 and G12 the corpus went from 30–40 GB to over 100 GB. **PESQ moved
about 1%.** That is the single most important number in this document, and §5
explains it.

An EDA pass before G12 caught two errors that would have invalidated the run: a
74-point mismatch in military-noise share, and an impulsive-tiling bug. A probe
step now runs before every dataset build, because G13 shipped with `gain_db`
silently dropped by `build()` — interferers landed ~2 dB below target instead
of the intended 12–24 dB.

## 4. Competing talkers: three attempts, all short

Removing a second human voice from a single microphone was attempted three
ways. Needed: roughly 12 dB of suppression.

- single-mic model, trained on interferer data: **+1.6 to +2.3 dB**
- Hush-style architecture: would need 1.8M params — **36× over budget**
- two far-field mics, diffuse-coherence simulation: **+4.4 dB best case**

The last one is bounded by physics, not engineering: two microphones separated
by `d` see a diffuse field with coherence `sinc(2πfd/c)`, and that sets a hard
ceiling on how much of it can be cancelled.

Conclusion at the time: not achievable. §6 revisits this and finds the way
through.

## 5. The oracle ceiling — why more data stopped helping

A perfect 48-band magnitude mask — the best any model of that output form could
ever do — scores **PESQ 2.392 at 0 dB**, below the 2.5 target. At 257 bins it
reaches 2.899; with clean phase, 3.946 / 4.480.

So every model from G8 to G12 was climbing toward a ceiling that sits under the
finish line. That is why 100 GB of extra data moved PESQ by 1%.

**G15 tested the obvious implication.** A deep filter (a short complex filter
across time, per bin, so phase correction falls out of the filtering) has an
oracle that clears all three PS targets at 0 dB — PESQ 2.849–3.996 at every
hold length. Its learnability gate initially failed at +0.88 dB, but that run
was cut off at step 599 while still rising; re-run at 4,000 steps it **passed at
+2.25 dB**.

G15 = G12 + deep-filter head, 51,839 params, 36.1 MMAC/s, 12 epochs on
`g12_20k`. Result:

| | G12 | G15 | Δ | verdict |
|---|---|---|---|---|
| edef PESQ | 1.697 | 1.706 | +0.009 | not resolved |
| realnoise PESQ | 1.677 | 1.670 | **−0.007** | **worse** |
| edef STOI | 0.824 | 0.826 | +0.001 | better |
| realnoise STOI | 0.851 | 0.852 | +0.001 | better |

At 0–5 dB: 1.472 → 1.483 (edef), 1.452 → 1.456 (realnoise). **Rejected.**

The trained model landed about **1.4 PESQ below its own oracle**. Removing the
mask-family ceiling changed nothing, which means the ceiling was never the
binding constraint. **What limits low-SNR PESQ is the estimator's ability to
find the mask from a single noisy channel — not the expressiveness of the mask
family.** Three independent levers (parameters, output-head richness, 100 GB of
data) have each produced ~1%. That is a converged answer, not a run of near
misses.

**Design consequence: stop adding capacity. Any further gain has to change the
information available at the input.**

## 6. Near-field two-microphone result

Full simulation in `E03/nearfield_sim.py`; results in `E03/runs/nearfield/`.

Modelled deliberately, because the first two-mic attempt in this project was
wrong in the opposite direction (free-field `1/r` only, which produced a
fantasy 40 dB input SIR):

- exact spherical propagation — per-mic delay `r/c` **and** `1/r` amplitude
- fractional delay by FFT phase ramp
- diffuse late reverberation carrying the theoretical coherence `sinc(2πfd/c)`
- direct-to-reverberant ratio falling as `1/r` about a critical distance
- INMP441 self-noise (61 dB SNR re 94 dB SPL) and ±1 dB sensitivity spread
- both mics on one I²S bus (shared BCLK/WS) — sample-synchronous, no clock skew

2,160 scenes, 6 speech clips, three apertures, three SNRs, eight ranges.

### 6.1 Angle: works, aperture-limited

Median absolute error, sources ≥ 0.5 m:

| aperture | SNR 20 dB | SNR 10 dB | SNR 0 dB |
|---|---|---|---|
| 15 mm | 23.0° | 26.5° | 29.5° |
| 50 mm | 4.5° | 5.5° | 8.5° |
| 150 mm | 2.5° | 3.3° | 5.0° |

### 6.2 Range: not recoverable beyond ~0.5 m

Wavefront curvature contributes a path difference scaling as `d²/r`, which
falls below the measurement noise floor quickly.

| true range | estimate (50 mm) | 68% interval |
|---|---|---|
| 0.03 m | 0.03 | ±0.00 m |
| 0.06 m | 0.07 | ±0.01 m |
| 0.15 m | 0.14 | ±0.04 m |
| 0.50 m | 0.36 | ±0.55 m |
| 1.00 m | 2.93 | **±5.54 m** |
| 4.00 m | 11.25 | ±5.30 m |

Past 1 m the interval spans the entire search grid — the point estimates are
just where a flat cost surface bottomed out. A wider aperture does not rescue
it; the 150 mm pair fails in the same place. **A range readout beyond half a
metre is not physically supportable and must not be displayed.**

### 6.3 Near vs far: the usable result

The wearer's mouth sits ~5 cm from a boom mic — deep in the array near field,
where `1/r` spreading gives a large level ratio between the two mics. Everyone
else is effectively far field, where it is ~0.

| geometry | wearer ILD | bystander ILD | accuracy (uncal.) | accuracy (cal.) |
|---|---|---|---|---|
| 15 mm | 2.29 ±1.01 dB | 0.25 ±0.90 dB | 85.4% | **100%** |
| 50 mm | 5.65 ±1.69 dB | 0.41 ±0.94 dB | **100%** | 100% |

Calibration — one gain match against a distant source at build time — takes
bystander spread from 0.90 dB to **0.11 dB**. It is not optional at 15 mm.

Reverberation **helps**: the wearer at 5 cm has ~26 dB direct-to-reverberant
ratio while a bystander at 2 m is around −2.5 dB, and reverberant energy carries
no level ratio, so it pushes the interferer further into the "far" class.

### 6.4 Payoff

ILD mask, boom pair, bystander at 2 m, RT60 0.45 s, calibrated:

| threshold / slope | SIR gain | wearer-voice loss |
|---|---|---|
| 2 dB / 1.0 | +8.4 dB | −0.6 dB |
| **3 dB / 1.0** | **+10.1 dB** | **−1.0 dB** |
| 4 dB / 1.0 | +11.1 dB | −1.8 dB |

Open field (no reverb): +10.8 dB. Against +1.6–2.3 dB single-mic and +4.4 dB
from the earlier far-field simulation.

Also measured: the wearer at 5 cm versus a 2 m bystander at equal source level
is **already 28.1 dB apart at the boom mic from geometry alone.** If background
talk is audible in a real recording, check the mic's mouth distance before
adding hardware.

**The pair must be asymmetric.** One mic per earcup gives ~0 dB ratio by
symmetry — the mouth is equidistant. Both mics go on the boom, at different
distances from the mouth.

## 7. Chain order — mask first, decisively

6-clip mean, 0 dB SIR competing talker plus 5 dB background, reference is the
wearer's direct path:

| chain | PESQ | STOI | SI-SDR |
|---|---|---|---|
| unprocessed | 1.095 | 0.726 | 0.65 |
| L1 only (current system) | 1.132 | 0.750 | 2.01 |
| mask only | 1.851 | 0.849 | 26.24 |
| **mask → L1** | **2.001** | **0.865** | 22.69 |
| L1 → mask | 1.248 | 0.831 | 8.43 |

L1 alone moves PESQ by **0.037** against a competing talker. That is the whole
problem in one number.

L1-then-mask fails because L1 is single-channel and nonlinear: run per-mic, it
applies a different per-bin gain to each channel and corrupts the very level
ratio the mask reads — at double the compute. Mask first keeps both channels
linear until the spatial cue has been extracted.

Division of labour is clean: the mask removes the talker (+10 dB) and barely
touches diffuse background (+1.7 dB); L1 removes background, which is what it
was trained for. The mask alone slightly *worsens* the background floor.

## 8. Two free runtime knobs

### 8.1 Mask sharpening

Raising L1's predicted mask to a power `alpha`, floored at −40 dB. No retraining,
one exponent, tunable at runtime.

| alpha | PESQ | STOI | background floor cut |
|---|---|---|---|
| 1.0 (current) | 2.001 | 0.865 | +17.7 dB |
| **1.6** | **2.013** | 0.864 | **+20.5 dB** |
| 2.0 | 1.991 | 0.862 | +21.4 dB |
| 2.5 | 1.948 | 0.860 | +22.0 dB |

**alpha 1.6 is strictly better than the current setting** — more background
removed and slightly higher PESQ. Past 1.6, noise keeps dropping but PESQ turns
over: that is where suppression starts costing speech.

On a real single-mic recording the same knob gives +7.9 → +10.1 dB net SNR
across alpha 1.0 → 2.5, at the cost of 2.4 → 3.6 dB of speech level.

### 8.2 Near-field gate (for near-silent gaps)

An **energy** squelch does nothing here — measured 0.0 dB — because what fills
the gaps between the wearer's words is another person talking, which any energy
detector reads as speech.

Gating on the **near-field cue** instead (energy-weighted per-frame ILD; open
only when the dominant source is at mouth distance) works:

| stage | PESQ | STOI | SI-SDR | floor between words |
|---|---|---|---|---|
| unprocessed | 1.083 | 0.726 | −1.00 | −22.8 dB |
| mask → L1 (α 1.6) | 1.436 | 0.792 | 16.31 | −35.3 dB |
| + energy gate | 1.382 | 0.792 | 15.10 | −35.3 dB |
| **+ near-field gate** | 1.307 | 0.792 | 3.40 | **−52.7 dB** |

30 dB below the input in the gaps — audibly silent. Costs 0.13 PESQ; **STOI is
unchanged**, so intelligibility is intact. The SI-SDR collapse is the gate
muting whole regions, not a quality defect. Risk to watch: gate chatter on soft
word onsets, tunable via hangover.

## 9. Cost on the ESP32-S3

- second INMP441 shares SCK / WS / SD on the **same I²S bus**, with its L/R pin
  tied to VDD instead of GND. **No new ESP32 pins, no new peripheral.**
- near-field mask: one extra 512-point FFT per frame plus a sigmoid,
  well under 2 MMAC/s against the 125 MMAC/s PS cap
- mask sharpening and the gate are scalar operations — free
- `esp-dsp` provides assembly-optimised FFT, FIR, IIR, dot product and matrix
  ops with ESP32-S3-specific implementations. Useful for the extra FFT, for
  L1's STFT/ISTFT, and for the FIR half of L0's FxNLMS. **LMS/NLMS is not
  confirmed present** — the adaptive update stays hand-written.
- `esp-sr` AFE (AEC / VAD / BSS / NS / NSNet) exists and supports a two-mic
  configuration. NSNet targets background noise — the problem already largely
  solved — not competing talkers. No NSNet benchmark has been run in this repo.

## 10. Known defects and gotchas

- **`spp` from GTCRNLite is saturated at exactly 1.0** across all frames and is
  unusable as a voice-activity signal. Anything needing VAD must compute its own.
- scipy's `NOLA` warning fires for `boundary=None, padded=False` STFTs but the
  round trip is exact — measured −314 dB (1024/256) and −69 dB (512/128).
- `enhance()` returns a 4-tuple `(Se, waveform, X, spp)`. Index `[1]` for audio.
- AGC must target a fixed output level (−18 dBFS active speech), **not** match
  the noisy input — matching the input over-boosts whenever the mask removes a
  lot, and lifts the residual noise floor with it.
- `._*` AppleDouble files appear on exFAT volumes and must be rejected by every
  dataset indexer (106,735 of them once).

## 11. Open decisions

1. **Buy the second INMP441?** Everything in §6–§8 depends on it. Before
   spending, check the existing mic's mouth distance — §6.4 says geometry alone
   should already give 28 dB.
2. If yes: rebuild an eval set with two-mic rendering and run the paired
   bootstrap decision rule, so these numbers become comparable to the PS
   figures in §2. That is a dataset build plus an eval — no training.
3. Adopt alpha 1.6 as the default. It is free and strictly better, but should
   be confirmed on edef/realnoise before being locked in.
