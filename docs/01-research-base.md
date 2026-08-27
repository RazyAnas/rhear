# RHEAR — Research Base
### Literature review for SIH PS 26052 (DRDO / DDP-iDEX): AI/ML-enabled adaptive noise cancellation

> Status: literature pass complete, 2026-08-27. Every numeric claim below is tagged
> `[v]` = verified against the source this session, or `[u]` = unverified / carried from
> prior notes and still to be confirmed against the primary source.

---

## 0. The single most important framing decision

The PS title says **"adaptive noise cancellation (ANC)"**. The PS success metrics say
**SNR > 15 dB, STOI > 0.85, PESQ > 2.5**.

Those two things are not the same problem, and conflating them is the most common way
this PS gets built wrong.

| | Active Noise Control | Speech Enhancement |
|---|---|---|
| Goal | Destructive acoustic interference at the **ear** | Statistical estimation of clean speech from a **microphone stream** |
| Output | Anti-noise sound pressure from a speaker | A cleaned digital signal for the radio/PTT |
| Metric | NMSE / dB attenuation at error mic | SI-SDR, STOI, PESQ |
| Latency budget | **acoustic causality: tens of µs** | perceptual: 10–40 ms |
| Physics | constrained by coherence + secondary path | constrained by information in the observation |
| Failure mode | instability, howl, amplification | over-suppression, musical noise, lost consonants |

STOI and PESQ are **intrusive, reference-based** measures. You cannot compute them on the
sound field inside an ear cup, because there is no aligned clean reference there. They are
computable only on a signal path — i.e. the communication (boom-mic → radio) path.

**Therefore PS 26052 is really asking for two co-designed subsystems in one headset:**

1. a **protection loop** (ANC + passive + impulse limiting) whose job is the wearer's ear, and
2. a **communication loop** (AI speech enhancement) whose job is the transmitted message,
   and which is what the STOI/PESQ/SNR numbers are scored on.

Everything in the RHEAR architecture follows from taking that split seriously. See
`02-architecture.md`.

---

## 1. What the physics says before any ML is involved

These four constraints bound the design. They are not negotiable and no neural network
removes them. Getting these right in the pitch is worth more than any model choice.

### 1.1 The causality constraint (kills naive "AI in the ANC loop")

In a feedforward ANC system the primary acoustic path carries noise from the reference mic
to the ear; the secondary path carries the anti-noise from the speaker to the ear plus the
whole electrical chain. Causal control requires

```
τ_primary  ≥  τ_secondary + τ_electrical
```

where `τ_electrical = ADC group delay + processing + DAC group delay + amp`. If this fails,
the controller can still cancel **narrowband or periodic** noise (it predicts the next
period) but **cannot cancel broadband random** noise. `[v]`

For a circumaural cup the outer reference mic to the ear is roughly 4–10 cm of air:

```
τ_primary ≈ 0.04–0.10 m / 343 m·s⁻¹ ≈ 120–290 µs
```

The secondary path (driver to ear, ~2 cm) is ~60 µs. That leaves the **entire electrical
chain a budget on the order of 60–200 µs.** A 16 kHz / 10 ms STFT frame is 10,000 µs — 50×
over budget before the network even runs. Sub-mm-scale systems (in-ear) are the ones where
causality becomes genuinely hard; metre-scale ducts are where it is easy. `[v]`

**Consequence:** a deep network can never sit *in* the sample path of the broadband
feedforward ANC loop. It must sit in the **coefficient path**. That is the central RHEAR
design decision, and conveniently it is exactly what the SFANC/GFANC/PFANC line of research
(§3) already validates.

**Second consequence, and a correction to our earlier plan:** you cannot demonstrate genuine
broadband ANC through a laptop + USB audio interface. Class-compliant USB audio round-trip
latency is 5–20 ms — 25–100× the causality budget. A laptop can host live *speech
enhancement* honestly, and can host ANC honestly **only for periodic noise** (engine, rotor
harmonics), where a delay of an integer number of periods is acoustically free. Broadband
cancellation (wind, blast tail) needs a microcontroller/DSP sitting at the mics. Plan the
demo around that truth rather than around a claim a judge with an acoustics background will
puncture in ten seconds.

### 1.2 The coherence bound (tells you where to put the mics)

The best possible attenuation of a single-reference feedforward controller at frequency `f`
is set by the ordinary coherence `γ²_xd(f)` between the reference signal `x` and the
disturbance `d` at the error point:

```
NR(f)  ≤  −10 · log₁₀ ( 1 − γ²_xd(f) )
```

| γ²_xd | max attenuation |
|---|---|
| 0.90 | 10.0 dB |
| 0.99 | 20.0 dB |
| 0.999 | 30.0 dB |

This is why *microphone placement is an algorithmic decision, not a mechanical one*, and why
adding a second reference mic (spatial diversity) can buy more dB than a bigger network.
A 2025 information-theoretic result generalises this: a unified lower bound on attainable
NMSE decomposes into an information term (how much disturbance entropy the anti-noise signal
captures) plus a support term (bands the actuator physically cannot reach), validated on
NOISEX. `[v]`

### 1.3 Impulsive noise is a documented failure mode of LMS, and the model says why

Gunshot and blast pressure are commonly **modelled** as symmetric α-stable (SαS) processes
with α < 2. Under that model the **second moment does not exist**, so FxLMS is minimising
`E[e²(n)]` — an undefined quantity — the gradient estimate has unbounded variance, and the
filter diverges. `[v]`

Two caveats worth stating precisely, because a panel will probe them:

- The divergence argument is rigorous **for the idealised SαS model**, not a proof about any
  particular real gunshot. Real impulses are finite-energy, band-limited by the cup and the
  mic, and often clipped before they reach the algorithm.
- What *is* empirically established, independent of the distributional assumption, is that
  conventional FxLMS becomes unstable or performs badly under impulsive disturbance — this is
  the motivating premise of the entire robust-ANC literature. `[v]`

So the α-stable model earns its place as the **reason to design for robustness**, not as a
claim about the data. The system is built around a detector, not around a distribution:

Known fixes, in order of increasing sophistication:
- **FxLMP**: minimise `E|e(n)|^p` with `p < α`. Needs α known. `[v]`
- **Threshold / clipped FxLMS**: clip reference and error before the update. `[v]`
- **FxlogLMS**: logarithmic transform of the error; stable *without* knowing α. `[v]`
- **M-estimate / score-matched** robust updates. `[v]`

```
              transient detector
                      │
              is this an impulse?
                      │
                 YES  │  NO
        ┌─────────────┴──────────────┐
        ▼                            ▼
  freeze / heavily damp        robust update rule
  adaptation, hold last-       (log-LMS by default)
  good filter                          │
        +                              │
  analog limiter + passive             │
        └─────────────┬────────────────┘
                      ▼
              stable controller
```

RHEAR uses the FxlogLMS-class update, with the detector-driven freeze as an *optional* extra.
Recent work on robust logarithmic and fractional-order filtered-X algorithms supports the
update side.

**Simulation has since sharpened both halves of this** (`05-digital-twin.md` §3):
- The divergence is real, but it belongs to **unnormalised** FxLMS. FxNLMS survives, because
  normalising by `‖x̂‖²` already divides out a loud reference.
- The freeze changed post-event attenuation by **at most 0.23 dB** across six conditions, so
  it is not what keeps the controller stable. The detector's real value is the limiter and
  `PROTECT` mode — protecting the ear, not the filter.

The physically decisive mechanism turned out to be **reference-microphone saturation**: at a
135 dB AOP against a 160 dB event the controller sees a clipped reference against an unclipped
error. That is a hardware-shaped problem, and it is why the attenuating acoustic port and the
pre-ADC analog limiter matter more here than the choice of cost function.

### 1.4 Hearing protection is a passive/analog problem, and honesty here is a scoring asset

- Gunshot peaks run **144 dB SPL (.22 rifle) to 172 dB SPL (.357 revolver)**; NIOSH's
  recommended ceiling for peak impulses is **140 dB SPL**, i.e. one shot can be a full
  day's allowable exposure. `[v]`
- Measured **impulse peak insertion loss (IPIL)** of real hearing protectors under
  ANSI S12.42: ≈ **17.1 dB** at 148–152 dB and ≈ **24.9 dB** at 166–170 dB. `[v]`
- **MIL-STD-1474E (2015)** dropped the plain peak-SPL criterion used in 1474D and evaluates
  impulse hazard with an energy metric plus the **AHAAH** electro-acoustic cochlear damage
  model. `[v]`

**We must never claim the AI cancels a gunshot.** The defensible claim is:
passive attenuation + analog/early-digital limiting provide the hearing protection; the AI
layer restores *intelligibility* through and after the event, and prevents the adaptive
controller from being destabilised by it. That framing survives contact with a DRDO
audiologist. The other one does not.

---

## 2. Speech enhancement: what to actually build on

### 2.1 The mechanisms worth stealing (not whole networks)

| Mechanism | Origin | Why it matters here |
|---|---|---|
| Complex-domain / phase-aware masking | DCCRN (NWPU, Interspeech 2020) `[u: 3.7M params, DNS-2020 RT track 1st]` | Heavy suppression without phase repair yields metallic speech and destroys stop consonants — fatal for call signs and coordinates |
| Decoupled magnitude + phase decoders | MP-SENet (Interspeech 2023) `[v: exists]` | Phase is not a residual; treating it as a parallel prediction target is measurably better |
| Full-band + sub-band fusion | FullSubNet (Westlake) `[u]`, S-DCCRN `[u]` | Defence noise is frequency-structured: engine = LF, rotor = harmonic comb, wind = LF broadband, blast = broadband transient. One scale cannot cover all |
| Grouped / factorised convolutions + subband feature extraction + temporal recurrent attention | **GTCRN** (ICASSP 2024) — **48.2 k params, 33.0 MMAC/s**; a second config reported at 23.7 k params / 39.6 MMAC/s `[v]` | This is the efficiency reference point for the whole project |
| Sub-band down/upsampling + dual-path RNN + a noise detector that skips computation | **LiSenNet** (ICASSP 2025) — **37 k params, 56 M MAC/s** `[v]` | Conditional computation: don't spend MACs on frames with no speech |
| Speech-presence-probability as an auxiliary task, losses weighted by homoscedastic uncertainty | Wang, Zhu & Kodrasi (arXiv 2011.07547, ICASSP 2021) `[v]` | Free regularisation; the shared encoder learns "where is speech" instead of only "what to output". Directly protects weak consonants |
| Deliberate residual-noise control instead of maximal suppression | Institute of Acoustics, CAS `[u]` | Encodes the correct objective: intelligibility beats dB |
| Dynamic noise-aware training over 100+ noise types | USTC `[u]` | Generalisation to unseen defence noise is the actual requirement |

### 2.2 Deployment evidence — the model *does* fit in a headset

This is no longer speculative, and it is the strongest feasibility argument available:

- A channel-grouped iterative CRN (**GTFCRN**) was **quantised to INT8 and deployed on a
  CSK6012 chip inside a noise-reducing headphone**, with a shell microphone capturing
  outside sound and the processed speech played to the wearer (J. Audio Speech Music Proc.,
  2026). `[v]` That is almost exactly the RHEAR communication path.
- RNN-based speech enhancement has been run on a multi-core MCU with mixed FP16/INT8
  post-training quantisation. `[v]`
- Sub-millisecond hearable enhancement: a 626 k-param LSTM predicts taps of a **minimum-phase
  FIR filter** applied sample-by-sample, giving **0.32–1.25 ms algorithmic latency,
  3.35 ms end-to-end, 4.1 dB SI-SDRi, 376 MIPS** on low-power DSP (arXiv 2409.18239). `[v]`
  Note the pattern: *the network emits filter coefficients; the filter does the sample work.*
  Same trick as the ANC side. RHEAR uses it twice.
- Hearing-aid-class targets for reference: 1–3 mW power, 512 KB–2 MB memory, 10 ms
  first-sample clinical latency threshold. `[v]`

### 2.3 Latency arithmetic for the communication path

With standard symmetric STFT, algorithmic latency = window length (you need the whole frame
before you can transform). With **asymmetric analysis/synthesis windows** you decouple
latency from frequency resolution: use a long analysis window (good resolution) with a short
synthesis window, so

```
L_alg = L_synthesis + hop     (not L_analysis + hop)
```

subject to the modified COLA condition on the *product* of the two windows. This is the
standard route to a 4–8 ms algorithmic latency SE front end that still has 32 ms of spectral
resolution. RHEAR's L1 loop is specified this way.

---

## 3. Active noise control: what to actually build on

### 3.1 The key line of work — neural *filter selection/generation*, classical *filtering*

This NTU Singapore (Shi, Luo, Gan et al.) line is, for our purposes, the most important
cluster in the field, because it is the only one that resolves the §1.1 causality problem
without hand-waving:

| Method | Idea | Verified? |
|---|---|---|
| **SFANC** (Signal Processing, 2021) | A CNN picks, per noise frame, the best filter from a bank of **pre-trained fixed control filters**. Rapid response, robust, no per-sample adaptation transient. | `[v]` |
| **Hybrid SFANC-FxNLMS** (arXiv 2208.08082) | A lightweight **1D CNN selects** the pre-trained filter for each frame, and **FxNLMS keeps updating** that filter's coefficients **at the sampling rate**. | `[v]` |
| **GFANC** (ICASSP 2023, arXiv 2303.05788) | *Generates* control filters by combining subband components of a single pre-trained broadband filter via a perfect-reconstruction filter bank — needs far less prior data than a filter bank per noise type. | `[v]` |
| **PFANC** (arXiv 2606.08171, 2026) | A **CRNN predicts the filter for the *next* frame** from several past frames, removing SFANC/GFANC's inherent tracking lag; justified with a high-order Markov argument; shown to transfer across acoustic paths. | `[v]` |
| **D-SFANC** (arXiv 2601.06981, 2026) | A CNN estimates **azimuth and elevation** of the noise source *and* picks the control filter. Better reduction with **shorter response time**, including under reverberation — prior direction studies were free-field only. | `[v]` |
| **PD-SFANC** (arXiv 2604.23144, 2026) | D-SFANC cannot track a **moving** source. A CRNN takes magnitude+phase spectrograms from J mics over K=4 context frames and **classifies next frame's DoA** on a 10° grid (**V=36 sectors**, one pre-trained FxLMS filter each), cross-entropy loss, **one frame = 0.5 s ahead**. Holds NRL > 15 dB under constant-rate and time-varying motion where D-SFANC lags a frame. **DoA accuracy > 90% only at SNR ≥ 20 dB.** | `[v]` |
| **SF-GFANC** (arXiv 2607.12807, 2026) | Multi-task CRNN jointly estimates **3D spatial cues (distance, elevation, azimuth)** and the sub-filter combination weights; robust to unseen rooms and noise types. | `[v]` |
| SFANC by **frequency-response matching in headphones** (Applied Acoustics, 2023) | Fixed-filter selection applied specifically to headphones rather than ducts/rooms. | `[v]` (title; full text paywalled) |

**Two things to carry over from the directional work, and two to leave behind.**
Carry over: (i) DoA is a *control-relevant* variable, not just a perception feature, and
(ii) predicting the next frame's spatial state beats reacting to the current one. Leave
behind: (i) PD-SFANC's **0.5 s frame** — a head turn at 100–300 °/s moves 50–150° in that
time, so a headset needs ~16 ms frames; and (ii) SF-GFANC's **3D distance/elevation** claims —
two reference mics on a head give azimuth and (with motion) front/back, not elevation or
range. See architecture §5.3.2–5.3.4.

**Read the hybrid SFANC-FxNLMS structure carefully — it is the RHEAR L0/L2 split:**
slow neural block chooses/creates *coefficients*; fast classical block runs at *sample rate*.
Latency of the network is completely decoupled from acoustic causality, because the network
is never in the audio path.

### 3.2 The other line — end-to-end Deep ANC

| Method | Idea | Verified? |
|---|---|---|
| **Deep ANC** (Zhang & Wang, OSU — Interspeech 2020 / Neural Networks 2021) | ANC as supervised learning: a CRN estimates the real/imaginary spectrogram of the canceling signal from the reference. Large-scale multi-condition training. **> 6 dB better NMSE than FxLMS** in untrained noise with nonlinear distortion. | `[v]` |
| Low-latency ANC with an **attentive recurrent network** (IEEE/ACM TASLP 2023) | Same family, latency-aware. | `[v]` |
| **Speech-preserving deep ANC in reverberant environments** (arXiv 2604.10979, 2026) | CRN+LSTM with complex spectral mapping and a **speech-retention loss** that suppresses noise while *keeping* target speech; ISM-simulated rooms; clearly beats FxLMS on non-stationary noise like babble. | `[v]` |
| **Deep Active Speech Cancellation** with a Mamba-masking network (arXiv 2502.01185) | Masking directly on the encoded reference + multi-band segmentation for phase alignment; **+7.2 dB (ANC) / +6.2 dB (ASC)**. | `[v]` |
| **DecNet-LMS** (arXiv 2511.03162) | Fixed-weight NN performs secondary-path inversion and crosstalk decoupling; an LMS filter models the primary path online. **Time-domain to preserve causality and avoid latency.** | `[v]` |
| Neural secondary-path modelling (ECUST; CAS path-decoupled ANC; BIT nonlinear ANC) | NN replaces the FIR secondary-path estimate to capture driver/amp nonlinearity. | `[u]` |

Note that **speech-preserving ANC** is a genuinely important find for a *defence* headset:
a wearer must still hear a squad-mate shouting. It is the ANC-side analogue of hear-through
and it is exactly the "situational awareness" feature tactical headsets sell on.

### 3.3 What conventional ANC actually achieves (baseline honesty)

- Feedback FxLMS in-ear prototype: **~15 dB, effective up to ~600 Hz**. `[v]`
- Well-tuned hybrid feedback portion: **~20 dB at ~150 Hz**. `[v]`
- Hybrid feedforward+feedback is standard practice for leaky/circumaural cups. `[v]`

So: **ANC gives you low frequencies. Passive gives you high frequencies. AI gives you the
message.** Say this on the first slide.

### 3.4 Hear-through / situational awareness constraints

- Desirable end-to-end transparency latency is **< 2 ms**, with **≤ 1 ms algorithmic**
  after hardware overhead; a working AR-audio headset measured **< 1.4 ms** throughput. `[v]`
- Leakage superposition of direct and processed sound causes **comb filtering**; all-pass
  based equalisation is a known mitigation. `[v]`
- Standard stereo passthrough degrades spatial hearing / localisation — a real operational
  cost in the field. `[v]`

---

## 4. Data

| Source | Contents | Fit |
|---|---|---|
| **MAD — Military Audio Dataset** (CC BY 4.0) | 7 classes: **communication, gunshot, footsteps, shelling, vehicle, helicopter, fighter**; **8,075 clips ≈ 12 h**, 16 kHz mono WAV, 1–10 s, 5 annotators, YouTube-sourced with games/fiction excluded `[v]` | The closest thing to a purpose-built defence noise corpus. Primary source for the noise classifier and for realistic mixing |
| **NOISEX-92** | machinegun, leopard (tank), M109, F16 cockpit, buccaneer jet, destroyer engine/ops room, factory, babble, white/pink, volvo `[v]` | Small but *genuinely military*, and it is the corpus the ANC literature benchmarks on — direct comparability |
| **DNS Challenge** (Microsoft, through DNS5/ICASSP 2023) | > 760 h clean speech incl. singing/emotion/non-English; noise from AudioSet + Freesound + DEMAND `[v]` | Clean-speech backbone and general noise variety |
| VoiceBank-DEMAND | the standard SE benchmark | Needed only for comparability of published PESQ/STOI numbers |
| AudioSet / ESC-50 / Freesound | ~2 M 10 s clips, ~600 event classes `[v]` | Mining extra gunshot/explosion/rotor/siren/wind material |
| **Our own recordings** | 3-mic rig, real rooms, real playback of defence noise at known SPL | Required — it is the only data that proves the *system*, not the model |

Gap to close: **impulse-response and secondary-path data for our own cup.** No public
dataset substitutes for measuring `Ŝ(z)` on the actual hardware.

---

## 5. Where the genuine research contribution sits

Everything above exists. Combining it is engineering. The parts of RHEAR that are, as far
as this literature pass shows, *not* already published as a unit:

1. **A single headset that runs neural-parameterised ANC (SFANC/GFANC/PFANC-class) and a
   lightweight streaming speech enhancer off one shared noise-scene encoder**, so that
   estimating the environment once drives *both* the control-filter choice and the
   enhancement conditioning. The two literatures (ANC, SE) barely cite each other.
2. **IMU-fused directional filter prediction.** The directional ANC literature treats the
   *source* as moving. On a headset the dominant relative-bearing change is the *listener*
   rotating, and a ₹100 gyroscope measures that directly — so the fast component of DoA
   prediction becomes analytic extrapolation instead of an audio inference problem that
   degrades below 20 dB SNR. Room-ANC systems have no head to instrument; this is available
   to us and, as far as this literature pass shows, unused.
3. **Direction used to bound the causal region, not only to select a filter.** Treating
   `Δ(θ) = τ_primary(θ) − τ_secondary` as a live quantity lets the controller *disengage* in
   bands where it would otherwise add uncorrelated energy. That is a safety property.
4. **α-stable-aware, detector-gated adaptation** as an explicit hearing-protection-integrated
   control mode, rather than as a robust-cost-function paper in isolation.
5. **An asymmetric, intelligibility-first loss** that penalises speech attenuation harder
   than noise retention, with the weighting *scheduled by the noise class* (aggressive under
   rotor, conservative under babble/speech-present).
6. A **published latency/coherence budget** for a ₹10–20 k build — i.e. treating cost as a
   first-class design constraint the way the papers treat MACs.

That is a defensible novelty claim: modest, true, and demonstrable in a hackathon.

---

## 6. Corrections to our earlier design notes

1. ~~"Run the AI + adaptive ANC on the laptop over USB"~~ — **impossible for broadband ANC**
   (§1.1). Valid for speech enhancement, and valid for periodic-noise ANC only.
2. ~~"4.3 ms < 10 ms therefore real-time"~~ — that is a *throughput* (real-time factor)
   argument. ANC needs a *causality* argument, which is 50× tighter. Both must be presented.
3. ~~"DCCRN as the deployment model"~~ — DCCRN is the right *mechanism* reference, wrong
   *size*. Deployment target is GTCRN/LiSenNet class: **10⁴–10⁵ params, 30–60 MMAC/s**.
4. ~~"SNR > 15 dB"~~ as a headline — must be disambiguated (§ metrics in `02-architecture.md`),
   and ranked below intelligibility.
5. The note's claim that neural secondary-path modelling should replace LMS *first* is
   backwards: measure `Ŝ(z)`, get classical FxNLMS stable and characterised, *then* show
   the neural version's delta. Otherwise there is no baseline to beat.

---

## Sources

- [GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources (ICASSP 2024)](https://ieeexplore.ieee.org/document/10448310/) · [official implementation](https://github.com/Xiaobin-Rong/gtcrn)
- [LiSenNet: Lightweight Sub-band and Dual-Path Modeling for Real-Time Speech Enhancement (ICASSP 2025)](https://ieeexplore.ieee.org/document/10888272/)
- [DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement](https://arxiv.org/pdf/2008.00264)
- [A Lightweight Hybrid Dual Channel Speech Enhancement System under Low-SNR Conditions (Interspeech 2025)](https://arxiv.org/abs/2505.19597)
- [A low parameter channel grouped iterative convolutional recurrent network for speech enhancement of noise-reducing headphones (GTFCRN, INT8 on CSK6012)](https://link.springer.com/article/10.1186/s13636-026-00455-4)
- [Towards Sub-millisecond Latency Real-Time Speech Enhancement Models on Hearables](https://arxiv.org/pdf/2409.18239)
- [Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with Mixed FP16-INT8 Post-Training Quantization](https://arxiv.org/abs/2210.07692)
- [Feasibility of Time-Domain DNN-Based Speech Enhancement on Embedded FPGA for Hearing Aid](https://arxiv.org/pdf/2606.04221)
- [Multi-task single channel speech enhancement using speech presence probability as a secondary task training target](https://arxiv.org/abs/2011.07547)
- [Selective fixed-filter active noise control based on convolutional neural network (Signal Processing)](https://www.sciencedirect.com/science/article/abs/pii/S0165168421003546)
- [A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control based on Deep Learning](https://arxiv.org/pdf/2208.08082)
- [Deep Generative Fixed-filter Active Noise Control (GFANC)](https://arxiv.org/abs/2303.05788)
- [Predictive Fixed-Filter Active Noise Control (PFANC) Using Convolutional Recurrent Neural Networks for Dynamic Noises](https://arxiv.org/pdf/2606.08171)
- [Directional Selective Fixed-Filter ANC Based on a CNN in Reverberant Environments (D-SFANC)](https://arxiv.org/abs/2601.06981)
- [Predictive Directional Selective Fixed-Filter ANC for Moving Sources via a CRNN (PD-SFANC)](https://arxiv.org/abs/2604.23144)
- [Spatial-Frequency Cued Generative Fixed-Filter ANC in Reverberant Environments (SF-GFANC)](https://arxiv.org/abs/2607.12807)
- [Selective fixed-filter active noise control based on frequency response matching in headphones (Applied Acoustics)](https://www.sciencedirect.com/science/article/abs/pii/S0003682X23003031)
- [Deep ANC: A deep learning approach to active noise control (Neural Networks 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0893608021001258) · [Interspeech 2020 version](https://www.isca-archive.org/interspeech_2020/zhang20i_interspeech.pdf)
- [Speech-preserving active noise control: a deep learning approach in reverberant environments](https://arxiv.org/abs/2604.10979)
- [Deep Active Speech Cancellation with Mamba-Masking Network](https://arxiv.org/pdf/2502.01185)
- [Active Noise Control Method Using Time Domain Neural Networks for Path Decoupling (DecNet-LMS)](https://arxiv.org/pdf/2511.03162)
- [Toward Optimal ANC: Establishing Mutual Information Lower Bound](https://arxiv.org/pdf/2505.17877)
- [Neural network-based ANC algorithms: a review](https://www.extrica.com/article/25037)
- [Review of Advances in Active Impulsive Noise Control with Focus on Adaptive Algorithms (Appl. Sci. 2024)](https://doi.org/10.3390/app14031218)
- [Improving robustness of filtered-x least mean p-power algorithm for α-stable impulsive noise](https://www.sciencedirect.com/science/article/abs/pii/S0003682X1100051X)
- [A survey on active noise control techniques — Part I: Linear systems](https://arxiv.org/pdf/2110.00531)
- [Causality study on a feedforward active noise control headset with different noise coming directions in free field](https://opus.lib.uts.edu.au/bitstream/10453/118176/4/Causality%20study%20on%20a%20feedforward%20active%20noise%20control%20headset%20with%20different%20noise%20coming%20directions%20in%20free%20field.pdf)
- [Active Noise Control: A Tutorial Review (Kuo & Morgan)](http://www2.coe.pku.edu.cn/tpic/2010913102917710.pdf)
- [Digital Augmented Reality Audio Headset (hear-through latency)](https://onlinelibrary.wiley.com/doi/10.1155/2012/457374)
- [Restoring Spatial Hearing in Active Noise Control Headphones (DAGA 2025)](https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/96.pdf)
- [A Military Audio Dataset for Situational Awareness and Surveillance (MAD)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11193796/)
- [NOISEX-92 database](http://mi.eng.cam.ac.uk/comp.speech/Section1/Data/noisex.html)
- [DNS Challenge repository](https://github.com/microsoft/DNS-Challenge) · [Interspeech 2020 DNS Challenge](https://arxiv.org/pdf/2005.13981)
- [MIL-STD-1474E](https://arl.devcom.army.mil/wp-content/uploads/sites/3/2022/09/ahaah-MIL-STD-1474E-Final-15Apr2015.pdf) · [MIL-STD-1474D](https://www.denix.osd.mil/soh/denix-files/sites/21/2022/12/03_MILSTD1474D-Noise-Limits.pdf)
- [The reduction of gunshot noise and auditory risk through the use of hearing protectors (CDC)](https://stacks.cdc.gov/view/cdc/111710/cdc_111710_DS1.pdf)
- [Firearms and Hearing Protection (peak SPL ranges)](https://hearingreview.com/hearing-loss/patient-care/evaluation/firearms-and-hearing-protection)
