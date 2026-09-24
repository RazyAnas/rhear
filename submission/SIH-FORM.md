# SIH SUBMISSION — PS 26052 — copy/paste fields

---

## 5. TECHNOLOGY BUCKET

**AI/ML, Cloud Computing, Blockchain**

(The problem statement is titled "AI/ML-enabled Adaptive Noise Cancellation".
"IoT and Electronics" also describes the prototype, but the PS asks first for a
model, a training framework and an inference engine — that is the AI/ML bucket.
Pick AI/ML.)

---

## 1. IDEA TITLE  — pick one, all under 100 characters

**Preferred (83 chars):**
RHEAR: Defence ANC where the AI tunes the canceller but never carries the audio

**Alternative A (89 chars):**
RHEAR: Edge AI noise cancellation that protects hearing and clears the radio separately

**Alternative B (74 chars):**
RHEAR: 50k-parameter ANC that beats a 25-million-parameter model on gunfire

---

## 4. ABSTRACT / SUMMARY  — ~2,900 characters, limit 10,000

RHEAR is an AI-driven adaptive noise cancellation system for defence headsets,
built and measured end to end on an ESP32-S3 — a microcontroller, not an
accelerator.

It starts from a requirement other systems collapse together. A soldier has two
acoustic problems that pull in opposite directions. He must not go deaf, which
means killing engine, rotor and gunfire noise. He must also be understood on the
radio, which means his voice and nothing else. Those goals conflict the moment a
comrade shouts a warning: that voice must reach his ears, because suppressing it
is a safety failure, and must not reach his radio, because the channel is his
alone. One microphone, one instant, two opposite targets. RHEAR treats protection
and communication as separate control problems with separate metrics.

The architecture follows from that. Three layers run at three rates on one chip.
L0 is hearing protection: FxNLMS, pure DSP at 48 kHz, a 20.8 microsecond
per-sample budget, and no neural network anywhere in it. L1 is the speech
enhancer for the radio: a 49,663-parameter model at 62.5 frames per second. L2 is
a scene engine at 62.5 Hz that emits coefficients, never audio. The AI therefore
sits in the coefficient path, not the audio path. The consequence is a safety
property: a stalled or failed model cannot open a hole in hearing protection,
because protection never waits on it.

Measured results. On a 300-clip defence evaluation set — gunshot, shelling,
helicopter, armoured vehicle, drone, fighter — RHEAR scores STOI 0.8236, PESQ
1.7159, SI-SDR 10.76 dB. Run under identical conditions, GTCRN's own published
weights score 1.6430 PESQ and 9.61 dB, and SepFormer, a 25.6-million-parameter
transformer, scores 5.56 dB. RHEAR beats a model 516 times its size by 5.2 dB
SI-SDR. On VoiceBank+DEMAND, the public benchmark the PS's thresholds come from,
it meets all three targets: SI-SDR 18.35 dB, STOI 0.936, PESQ 2.524.

The training pipeline generated over 100 GB of noisy-clean pairs across 251
speakers and 235 noise classes, with measured room impulse responses, clipping,
microphone response randomisation and competing talkers at controlled
signal-to-interference ratio. Every architectural change was accepted or rejected
by a paired bootstrap over 10,000 resamples on two independent evaluation sets.
Four of the team's own models were rejected under that rule, including one that
improved PESQ but degraded intelligibility.

The project also reports its own ceiling. Scoring the ideal mask shows that a
perfect 48-band magnitude mask cannot exceed PESQ 2.392 at 0 dB against a 2.5
target — so at low SNR the limit is the information in one microphone, not the
size of the network. That measurement redirected the work from bigger models to
better inputs, and it is why a second boom microphone, which separates the
wearer's voice from a bystander's at 100 percent using wavefront curvature alone,
is the next hardware step rather than a larger model.

---

## 2. IDEA DESCRIPTION  — ~11,500 characters, limit 50,000

### 1. THE PROBLEM, STATED PRECISELY

In a vehicle bay, a cockpit or a gun line, a soldier faces two acoustic problems
that are usually treated as one and are in fact opposites.

The first is protection. Engine, rotor, gunfire and shelling must be attenuated
or he loses his hearing. The second is communication. His voice must reach the
radio intelligibly, and nothing else should.

These conflict. If a comrade three metres away shouts a warning, that voice must
reach the soldier's EARS — suppressing it is a safety failure, not a feature. The
same voice must NOT reach his RADIO, because the channel belongs to him alone.
One microphone, one instant, two opposite requirements.

Conventional systems, and most deep-learning speech enhancers, optimise a single
"remove the noise" objective and therefore cannot satisfy both. RHEAR is built
around the separation.

### 2. ARCHITECTURE — THREE RATES, AND THE AI OUTSIDE THE AUDIO PATH

Three layers run concurrently on one ESP32-S3.

**L0 — protection.** FxNLMS adaptive cancellation at 48 kHz. Pure DSP. A 20.8
microsecond budget per sample. No neural network is in this loop.

**L1 — communication.** A causal neural speech enhancer at 62.5 frames per
second, 16 ms hop, feeding the radio.

**L2 — scene.** A scene engine at 62.5 Hz that classifies the acoustic
environment and emits COEFFICIENTS that retune L0. It never carries audio
samples.

The design decision the whole system rests on is that the AI lives in the
coefficient path, not the audio path. It tunes the canceller; it never conveys
the sound.

This is a safety property, not an optimisation. Any system that places a network
inside the cancellation loop has a failure mode in which a stalled model, a cache
miss or a dropped frame becomes a hole in the soldier's hearing protection.
RHEAR is architecturally incapable of that, and this is demonstrable live: the AI
core can be killed at runtime and cancellation continues uninterrupted.

The same split answers the ear-versus-radio conflict. The ear path removes
machine noise and PRESERVES human voices, so a shouted warning is heard. The
radio path removes machine noise AND every voice that is not the wearer's.

### 3. THE MODEL

GTCRNLite: 49,663 parameters, derived from a published ultra-low-complexity
architecture and adapted for this task and this silicon.

- Dual representation: a sub-band path over 96 ERB bands for local structure, and
  a parallel full-band path over all 257 STFT bins for global structure, fused
  before the recurrent stage. The PS asks for both; the model computes both.
- Encoder, dual-path RNN, decoder, causal in time so it can stream.
- Speech-presence head trained as an auxiliary task.

Compute and memory, against the PS's own caps:

| quantity | RHEAR | PS cap | headroom |
|---|---|---|---|
| int8 weight storage | 97.4 KB | 200 KB | 2.05× |
| compute | ~26–32 MMAC/s | 125 MMAC/s | ~4× |

Against one ESP32-S3 core at roughly 200 MMAC/s, L1 occupies about 15 percent.
Neither memory nor compute is the binding constraint.

### 4. DATASET PIPELINE

Over 100 GB of noisy-clean pairs were GENERATED rather than collected, which is
what makes the corpus scalable and the labels exact.

- 251 speakers; 235 noise classes; 4-second clips at 16 kHz.
- SNR drawn uniformly from −10 to +20 dB; one to three noise layers per clip.
- Defence noise by share of layers: gunshot 16.0%, armoured vehicle 10.4%,
  helicopter 9.8%, shelling 9.4%.
- Real MEASURED room impulse responses, applied with probability 0.6.
- Twelve augmentations. The three the PS names — random noise mixing,
  reverberation, clipping — plus microphone frequency-response randomisation,
  microphone self-noise, preamplifier nonlinearity, time-varying noise
  trajectories, gain jitter, and competing talkers at an explicitly controlled
  signal-to-interference ratio of 12–24 dB.

One detail that matters for impulsive noise: impulsive classes are never tiled.
A gunshot is placed once in a clip, so the model cannot learn an artificial
repetition rate that would not exist in the field.

### 5. TRAINING FRAMEWORK

Loss: SI-SNR + 30 × compressed magnitude + 15 × complex real/imaginary + 0.5 ×
speech-presence, with a multi-resolution STFT perceptual term available.

The magnitude term is deliberately ASYMMETRIC, with rho = 8: removing speech is
penalised eight times harder than leaving noise. Deleting a word from a radio
call is worse than passing some noise through it, and the loss says so.

Optimiser AdamW, learning rate 5e-4, cosine annealing, gradient clipping 5.0.

**The decision rule is the part worth attention.** A change ships only if PESQ
improves AND STOI does not regress AND SI-SAR does not regress, on BOTH
independent evaluation sets, judged by a paired bootstrap over 10,000 resamples
on identical clips.

Four of our own models were rejected under that rule:

| run | change | PESQ | verdict | why |
|---|---|---|---|---|
| g8 | 48 → 96 ERB bands | 1.6782 | kept | PESQ better |
| g9 | +56% parameters | 1.6846 | **rejected** | STOI significantly worse |
| g10 | asymmetry rho 8 → 4 | 1.6779 | **rejected** | STOI significantly worse |
| g11 | 26 → 251 speakers | 1.6945 | kept | PESQ better |
| g12 | 9 → 236 noise classes | 1.6969 | kept | other metrics, both sets |
| g13b | competing talkers | 1.7159 | kept — shipping | PESQ better |
| g15 | deep-filter output head | 1.7057 | **rejected** | PESQ worse on realnoise |

g9 GAINED PESQ and was still rejected, because intelligibility fell. A favourable
headline number was not enough. That is the rule doing its job.

### 6. RESULTS

**Against published models, on our own defence noise.** Same 300-clip set, same
scoring, same conditions.

| model | parameters | STOI | PESQ | SI-SDR |
|---|---|---|---|---|
| **RHEAR** | **49,663** | **0.8236** | **1.7159** | **10.76 dB** |
| GTCRN (DNS3 weights) | 23.7k | 0.8182 | 1.6430 | 9.61 dB |
| GTCRN (VCTK weights) | 23.7k | 0.7648 | 1.4442 | 6.45 dB |
| SepFormer | 25,613,569 | 0.8021 | 1.5963 | 5.56 dB |
| MetricGAN+ | ~2M | 0.7546 | 1.8197 | **−0.46 dB** |

RHEAR beats the architecture it derives from, using that architecture's own
published weights, on all three metrics. It beats SepFormer — 516 times larger —
by 5.2 dB SI-SDR.

We report the row that beats us, because it makes the methodological point.
MetricGAN+ posts the table's best PESQ and a NEGATIVE SI-SDR: it is trained
directly against PESQ and degrades the signal while scoring well on the metric it
optimises. This is precisely why our decision rule requires three metrics on two
sets rather than one number.

**Against the PS thresholds.** On VoiceBank+DEMAND, the public benchmark those
thresholds come from, 824 clips, after in-domain fine-tuning:

| metric | achieved | target | |
|---|---|---|---|
| SI-SDR | 18.35 dB | > 15 dB | MET |
| STOI | 0.936 | > 0.85 | MET |
| PESQ | 2.524 | > 2.5 | MET |

On our harder defence set, all three targets are met above 15 dB input SNR
(PESQ 2.643, STOI 0.962, SI-SDR 20.02 dB). Below that, STOI holds down to about
5 dB and PESQ does not. Section 7 explains why, because we measured it.

### 7. WE MEASURED OUR OWN CEILING

Rather than assume more training would close the low-SNR gap, we scored the
ORACLE: the ideal mask, computed from the clean speech the model never sees.

**A PERFECT 48-band magnitude mask scores PESQ 2.392 at 0 dB. The target is 2.5.**

The model was not underperforming. It was climbing toward a ceiling that sits
below the finish line. Three independent levers confirmed it:

- +56% parameters → +0.001 PESQ on training data
- a richer output head → landed 1.4 PESQ below its own oracle
- 30 GB → 100+ GB of data → about 1%

So at low SNR the limit is the INFORMATION IN ONE MICROPHONE, not the size of the
network. We stopped adding capacity and changed the input instead.

### 8. WHAT CHANGING THE INPUT BUYS — SIMULATED, NOT ASSUMED

A second microphone on the same boom, a few centimetres further from the mouth,
was evaluated in a 2,160-scene physical simulation modelling exact spherical
propagation, diffuse-field coherence, microphone self-noise and part tolerance.

The wearer's mouth sits ~5 cm away — inside the array's near field, where
spherical spreading gives a level ratio of about 5.7 dB between the two
microphones. Anyone beyond half a metre gives about 0.2 dB. That is a thirty-fold
difference in the cue, and it yields:

- **100% separation** of wearer from bystander, calibrated
- **+10.1 dB** interferer suppression for 1.0 dB of the wearer's own voice
- no voice enrolment, no training on the user, and it still works when he shouts

The same simulation returned an honest negative: absolute RANGE beyond about half
a metre is NOT recoverable. Wavefront curvature dies as d²/r, and at one metre the
confidence interval spans the entire search grid. A wider array does not help. We
therefore do not claim a range readout.

Cost: the second microphone shares SCK/WS/SD on the same I²S bus with its L/R pin
strapped opposite — no new pins, no new peripheral, under 2 MMAC/s.

### 9. PROTOTYPE AND CURRENT HARDWARE STATE

Running on an ESP32-S3-N16R8 with an INMP441 MEMS microphone and a MAX98357A
class-D amplifier:

- Live capture, processing and playback at 16 kHz, verified on hardware.
- A NEURAL voice activity detector (vadnet1_medium, from Espressif's esp-sr)
  running ON THE CHIP, deciding when the channel carries a voice.
- A gate that closes to DIGITAL ZERO — measured −120 dBFS — when no voice is
  present, so the radio channel is silent rather than quietly hissing.
- FxNLMS validated at +12.6 dB cancellation against a host reference.
- Live telemetry: microseconds per frame, real-time factor, achieved sample rate,
  and microphone format validation, all measured on the silicon.

**Stated plainly, because it matters:** L1's neural model runs on a host today.
Its on-chip port is staged and in progress — the bidirectional GRU, the largest
and most error-prone block, is written in C and validated against PyTorch to
−133.5 dB relative error. The encoder, full-band branch and fusion are ported and
validated to 2e-5. The decoder, ERB inverse and on-chip STFT are not yet written.
Memory and compute are not the obstacle; implementation time is.

The active cancellation path additionally requires a low-latency codec
(ADAU1772), because L0's causality budget is 146 microseconds, computed from a
7 cm reference-to-ear geometry against the standard causality condition. The
schematic is complete and electrically verified — 37 parts, 30 nets, ERC clean.

### 10. WHAT IS GENUINELY NEW

1. **Protection and communication as separate control problems.** The ear keeps
   human voices; the radio removes them. One microphone, opposite targets.
2. **AI in the coefficient path.** Hearing protection cannot be taken down by a
   model stall, and this is demonstrable by killing the AI core live.
3. **Near-field wavefront discrimination.** Telling the wearer from a bystander
   using the curvature of the sound wave — physics, not enrolment.
4. **Measuring our own ceiling.** We can state the score a perfect version of our
   model would achieve, which is how we knew to stop training and change the
   physics instead.

### 11. DEPLOYMENT

Defence vehicle crews, aircrew, artillery and gun-line teams, and equally
aerospace ground crew and high-noise industrial operations. The compute target is
a commodity microcontroller, not an embedded GPU, so unit cost and power are
compatible with issuing the system widely rather than to a few platforms.
