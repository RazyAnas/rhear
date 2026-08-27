# RHEAR — System Architecture
### Robust Hearing Enhancement & Active Reduction
PS 26052 · DRDO / Dept. of Defence Production–iDEX · v0.1 · 2026-08-27

Read `01-research-base.md` first; this document assumes its four physical constraints
(causality, coherence, α-stability, passive-protection primacy).

---

## 1. Design axioms

**A1. Neural computation is kept out of the ultra-low-latency ANC control loop.**
There the model acts through the *coefficient* path, so neural latency is decoupled from
acoustic causality. (Justified by §1.1 of the research base; validated by hybrid SFANC-FxNLMS,
GFANC and PFANC.)

This is a statement about **one loop's latency budget, not a blanket rule about AI in audio.**
On the communication path (L1) a small network processes audio samples directly and that is
entirely appropriate — its deadline is ~15 ms, and an INT8 model of this class has already
been deployed on-chip inside a headphone (GTFCRN on CSK6012). The rule is: *match the
computation to the loop's deadline.* Microseconds → deterministic DSP. Milliseconds → neural.

**A2. Two loops, two objectives, two clocks.**
Protection (the ear) and communication (the message) are separate control problems with
separate metrics. They share *perception*, not *computation*.

**A3. Intelligibility dominates attenuation.**
The loss function, the metric hierarchy, and the demo all rank speech preservation above dB
of suppression. A system that scores +20 dB SNR and drops the word "fire" has failed.

**A4. Passive first, analog second, classical third, neural fourth.**
Each layer only handles what the layer below could not. This is what makes a ₹15 k build
credible and a 33 MMAC/s model sufficient.

**A5. Everything claimed is measured on our hardware.**
Model metrics on public sets prove the model. Only rig measurements prove the system.

---

## 2. Three-tier control hierarchy

The whole architecture is one idea: **three loops running at three rates, coupled only
through low-bandwidth state.**

```
                                OUTSIDE ACOUSTIC FIELD
                       engine · rotor · wind · siren · blast · speech
                                          │
      ┌───────────────────────────────────┼───────────────────────────────────┐
      │                    PASSIVE CUP  (−20…−35 dB above ~1 kHz)             │
      └───────────────────────────────────┼───────────────────────────────────┘
                                          │
   ┌──────────────┬───────────────────────┼───────────────────────┬──────────────┐
   │  ref mic L   │   ref mic R           │            error mic  │   boom mic   │
   │  x_L(n)      │   x_R(n)              │            e(n)       │   z(n)       │
   └──────┬───────┴──────────┬────────────┴───────────┬───────────┴──────┬───────┘
          │                  │                        │                  │
          ▼                  ▼                        ▼                  ▼
 ╔════════════════════════════════════════════════════════════════╗   ╔═══════════════╗
 ║  L0  PROTECTION + CONTROL          f_s = 48 kHz, 1 sample      ║   ║  L1  COMMS    ║
 ║      ───────────────────────       ≈ 20.8 µs budget            ║   ║  16 kHz       ║
 ║  · impulse detector D(n)  · peak limiter                       ║   ║  4 ms hop     ║
 ║  · hybrid FF+FB FIR controller  w(n) ∈ ℝ^L                     ║   ║  ≤ 8 ms alg.  ║
 ║  · robust (log/p-norm) FxNLMS coefficient update               ║   ║               ║
 ║  · secondary-path model Ŝ(z)                                   ║   ║  streaming    ║
 ║  · hear-through mixer                                          ║   ║  neural SE    ║
 ║          ── pure DSP. No neural network in this path. ──       ║   ║  on z(n)      ║
 ╚═══════════▲══════════════════════════════════════╦═════════════╝   ╚═══▲═══════╤═══╝
             │ w₀, μ, leak, mode                    ║ y(n) anti-noise      │ γ, θ   │ ŝ(n)
             │ (updated every 16 ms)                ▼                      │        ▼
 ╔═══════════╩══════════════════════════════════════════════════════════════╩═══╗  radio
 ║  L2  NOISE SCENE ENGINE            frame rate 62.5 Hz (16 ms), lookahead 0   ║   PTT
 ║      ───────────────────                                                     ║
 ║  shared encoder φ(·) over [x_L, x_R, e] log-mel + subband features           ║
 ║    ├─ head A: noise class posterior  p(c | ·)   (7 defence classes + none)   ║
 ║    ├─ head B: control-filter generator  α(t) → w₀(t)  (GFANC/PFANC basis)    ║
 ║    ├─ head C: adaptation scheduler  (μ, leakage, robust-cost p, freeze)      ║
 ║    ├─ head D: SE conditioning vector γ(t), θ(t)  → FiLM into L1              ║
 ║    └─ head E: speech-presence probability / hear-through gate                ║
 ║          ── neural, ~10–15 MMAC/s, latency irrelevant to causality ──        ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
```

| Tier | Rate | Deadline | Nature | What breaks if it is late |
|---|---|---|---|---|
| **L0** | 48 kHz | 20.8 µs / sample | fixed-point FIR DSP | ANC becomes non-causal → broadband cancellation fails, possible howl |
| **L1** | 250 Hz (4 ms hop) | ≤ 8 ms algorithmic | streaming NN | transmitted speech feels delayed; radio protocol timing |
| **L2** | 62.5 Hz (16 ms) | ~100 ms soft | NN | controller adapts a little slower to a scene change — *graceful* |

**This table is the feasibility answer to "how does laptop AI run live on a headset".**
Only L0 has a hard microsecond deadline, and L0 is a few hundred MACs of FIR arithmetic that
a ₹300 MCU does easily. The neural parts sit behind soft deadlines and can be scheduled,
batched, quantised, and even dropped for a frame without instability.

---

## 3. Signal model and notation

Discrete time `n` at `f_s`. Frames `t`. Vectors bold-implicit.

```
Reference (outer) mics    x_L(n), x_R(n)            → x(n) ∈ ℝ^M , M = 2
Error (in-cup) mic        e(n)
Boom (speech) mic         z(n) = s(n) + v(n)        s = speech, v = leaked noise
Anti-noise output         y(n)
Primary   path (ref→ear)  P(z)      d(n) = (p * x)(n)
Secondary path (spk→ear)  S(z)      estimate Ŝ(z)
Control filter            W(z), taps w ∈ ℝ^L
```

Residual at the error mic:

```
e(n) = d(n) − (s * y)(n) + u(n)          u = uncorrelated / self-noise
y(n) = wᵀ(n) x(n)                        (feedforward part)
```

Optimal Wiener control filter in the frequency domain:

```
W_opt(f) = P(f) / S(f)
```

which is realisable only if `P` is "longer in delay" than `S` (causality, §1.1) and only to
the extent `x` is coherent with `d` (§1.2). Both are engineering targets, not afterthoughts.

---

## 4. L0 — Protection and control (48 kHz, no neural network)

### 4.1 Passive + analog front (the layer that does the heavy lifting)

- Sealed circumaural cup, target NRR ≥ 24 dB, most of it above 1 kHz.
- Analog compression/limiting **before** the ADC so a 165 dB transient never clips the
  converter. Once the ADC saturates, information is destroyed and no network recovers it.
- Headroom plan: mics rated ≥ 130 dB AOP, with an attenuating acoustic port for the outer
  pair so the blast peak lands inside the mic's linear range.

### 4.2 Impulse detector `D(n)` — cheap, sample-domain, hysteretic

Three statistics on a short sliding window `N_w` ≈ 64 samples (1.3 ms):

```
short-term energy      E(n)  = Σ_{k=0}^{N_w−1} x²(n−k)
onset slope            R(n)  = 10·log₁₀( E(n) / E(n−N_w) )
crest factor           C(n)  = max|x| over window  /  sqrt(E(n)/N_w)
spectral flatness      F(n)  = geometric-mean(|X|) / arithmetic-mean(|X|)   (from L2, lagged)
```

Impulse declared when `R(n) > R_th` (typ. 15–20 dB) **and** `C(n) > C_th` (typ. 6–8),
with `F(n)` used to reject loud-but-tonal events (a siren rises fast but is tonal).
Rationale from the gunshot-detection literature: gunfire is distinguished by *abrupt onset*
and *lack of tonality*, not by loudness alone. Release with hysteresis over ~200 ms so the
blast tail and early reflections are covered.

`D(n) ∈ {0,1}` drives three things simultaneously: the limiter, the adaptation freeze, and
an L2 event flag.

### 4.3 Hybrid feedforward + feedback controller

```
y(n) = w_ffᵀ(n)·x(n)  +  w_fbᵀ(n)·ê(n)
```

Feedforward handles what the reference mics can see (external, coherent). Feedback handles
leakage and cup-fit variation — the standard remedy for acoustically leaky/circumaural cups,
worth ~20 dB around 150 Hz on its own. The feedback branch reconstructs an internal
disturbance estimate `ê(n) = e(n) + (ŝ * y)(n)`.

Filter length: at 48 kHz, `L = 256` taps ≈ 5.3 ms of impulse response — enough for the cup's
acoustics, and 256 MACs/sample ≈ 12.3 MMAC/s per channel. Trivial.

### 4.4 Coefficient update — robust FxNLMS with α-stable protection

Baseline (Gaussian-ish noise, `D(n)=0`):

```
x̂(n) = (ŝ * x)(n)                               filtered reference

                     μ
w(n+1) = (1−λ)w(n) + ───────────────── · x̂(n) · ψ(e(n))
                     δ + ‖x̂(n)‖²
```

with leakage `λ` for numerical robustness and `δ` for regularisation. The **score function
`ψ(·)`** is what makes this impulse-safe:

| mode | ψ(e) | when |
|---|---|---|
| NLMS | `e` | stationary noise (engine idle, cabin) |
| log-LMS | `e / (ε + \|e\|)` | unknown/heavy-tailed — the **default**, stable without knowing α |
| p-power | `sign(e)·\|e\|^{p−1}`, `p < α` | when α has been estimated for the scene by L2 |
| frozen | `0` | `D(n) = 1` — adaptation halted, last-good `w` held. **Optional**: measured benefit ≤ 0.23 dB (`05-digital-twin.md` §3) |

Why this matters (§1.3): for SαS noise with α < 2, `E[e²]` does not exist, so plain FxLMS has
an undefined objective and diverges. Confirmed in simulation — see `05-digital-twin.md` §3,
where unnormalised FxLMS diverges unrecoverably under a 160 dB SPL burst.

**Two corrections from that experiment, both against our own earlier claim:**

1. **Normalisation is itself an impulse defence.** Dividing the update by `‖x̂‖²` divides out
   a loud reference. FxNLMS did *not* diverge; only the unnormalised form did. The divergence
   result applies to FxLMS, and we should say exactly that.
2. **The `frozen` mode does not earn its place as an adaptation-stability mechanism.**
   Ablated across step sizes and burst counts, the freeze changed post-event attenuation by
   **at most 0.23 dB**. Normalisation plus the log score function already carry it.

The detector is retained — it drives the analog limiter, `PROTECT` mode, hear-through ducking
and event logging, all of which concern the ear rather than the filter. But the freeze is
demoted from "belt and braces" to *optional*, and must not be presented as the thing that
keeps the controller stable. The braces were doing all the work.

**Stability envelope.** With `x̂` the filtered reference and `Δ` the secondary-path delay in
samples, convergence needs approximately

```
0 < μ < 2 / (L · σ²_x̂ · (Δ + 1))
```

and phase error in `Ŝ` must stay within ±90° over the control band, or the "cancellation"
adds energy. `Ŝ` is therefore measured, not assumed — see 4.5.

### 4.5 Secondary-path identification `Ŝ(z)`

Offline, at build time: white/log-swept-sine excitation through the driver, capture at the
error mic, estimate `Ŝ` by LS/RLS. Online: low-level (−40 dBFS) uncorrelated pilot noise
injection with a slow parallel LMS to track fit/temperature drift.

Nonlinearity in the driver+amp is real, and is the documented reason classical ANC becomes
unstable at high levels. Two-stage plan, in this order:
1. Linear FIR `Ŝ` + measured coherence and phase error. **Baseline, must exist first.**
2. A small neural `Ŝ_θ` (DecNet-style fixed-weight inversion, or a memoryless-polynomial +
   FIR Hammerstein model) evaluated as a *delta over that baseline*.

Doing (2) before (1) leaves nothing to compare against — a mistake worth avoiding.

### 4.6 Modes (selected by L2, executed by L0)

| Mode | Trigger | Behaviour |
|---|---|---|
| `PROTECT` | `D(n)=1` or SPL > threshold | limiter hard, adaptation frozen, hear-through ducked |
| `ANC_TONAL` | periodicity `P` high, predictability `H` high | narrowband/harmonic-targeted filter, long `L`, small `μ` |
| `ANC_BROAD` | `P` low, `H` low, broadband LF energy | broadband filter, larger `μ`, leakage up |
| `TRANSPARENT` | external speech detected (head E) | ANC relaxed in 300 Hz–4 kHz, hear-through path opened |
| `COMMS_PRIORITY` | PTT active | ANC held at last-good filter, all adaptation frozen for stability |

The `TRANSPARENT` mode is the speech-preserving-ANC idea from the 2026 reverberant deep-ANC
work, implemented as a *scheduled band-limited relaxation* rather than as a network in the
loop — same operational benefit, none of the latency cost.

---

## 5. L2 — Noise scene engine (the shared brain, 62.5 Hz)

One encoder, five heads. This is where RHEAR's integration claim lives: the same scene
representation drives *both* the controller and the enhancer.

### 5.1 Shared encoder φ

Input per 16 ms frame, over the last 8 frames:
- 64-band log-mel of the reference pair
- subband energy ratios, crest factor, spectral flatness
- **binaural cues**: inter-reference phase difference `∠(X_L X_R*)` and level difference
  `20log₁₀|X_L/X_R|` per band, plus coherence `γ²_LR(f)` (8 bands)
- **head angular rate `ω(t)`** from a MEMS IMU (see 5.3.2)

Backbone: depthwise-separable 1D convolutions → single GRU(64). Target ≤ 15 MMAC/s.

```
h(t) = φ( X(t−7:t) ) ∈ ℝ^64
```

### 5.2 Head A — acoustic **state** estimation (not just a class label)

A closed-set classifier is brittle exactly where the PS is hardest — unseen noise — and a
label is the wrong interface anyway. `rotor` tells the controller nothing actionable;
*"highly periodic, f₀ ≈ 22 Hz with six strong harmonics, highly predictable, low
impulsiveness, coherent across both reference mics"* tells it everything.

Head A therefore regresses a **continuous acoustic state vector** `a(t)`, with the class
posterior demoted to an auxiliary training target that regularises the encoder and gives the
demo a human-readable label:

| descriptor | estimator | what it drives |
|---|---|---|
| fundamental + harmonic set `f₀, {k·f₀}` | cepstral / autocorrelation peak, refined by the net | band allocation of the control filter; harmonic-targeted mode |
| **periodicity** `P ∈ [0,1]` | normalised autocorrelation peak height | tonal vs broadband control strategy |
| **impulsiveness** `I` | frame kurtosis + crest factor + onset slope | freeze policy, limiter threshold, loss ratio ρ |
| **predictability** `H ∈ [0,1]` | `1 −` normalised one-step LPC prediction error | how much attenuation is physically available, **and how much latency the loop can tolerate** |
| **bearing** `θ`, its rate `θ̇`, and `γ²_LR(f)` | inter-reference phase/level difference + IMU angular rate | causality margin `Δ(θ)`, per-ear filter selection, uncontrollable-band declaration (§5.3.3) |
| broadband level, per-band SNR | short-term spectra | mode thresholds, hear-through gain |
| **confidence** `κ` | predictive entropy / MC-dropout variance | low `κ` → fall back to the conservative broadband fixed filter |

**Why `predictability` is the most important number here.** §1.1 of the research base says the
causality constraint only bites for *unpredictable* noise: for a signal that is periodic with
period `T`, a processing delay of `kT` is acoustically free, because the controller can
synthesise the anti-noise for an event it has already seen a period ago. So `H` is not a
descriptive statistic — it is the variable that tells the scheduler how much of the electrical
delay budget it is allowed to spend, and it is why engine and rotor noise are tractable on
hardware that could never cancel wind.

Mapping from state to behaviour (executed by L0's mode selector, §4.6):

```
high P, low f₀, high H          →  aggressive low-frequency / harmonic ANC, long filter, small μ
strong harmonic comb, high H    →  harmonic-targeted fixed filter + slow FxNLMS refinement
broadband LF, low H  (wind)     →  conservative ANC, lean on passive + L1 speech enhancement
high I  (impulse)               →  PROTECT: freeze adaptation, limiter, passive; recover speech after
external speech present         →  TRANSPARENT: relax ANC in 300 Hz–4 kHz, open hear-through
low κ                           →  conservative fixed filter, no generated filter applied
```

This is what makes the phrase "adaptive ANC" mean something specific rather than being a
synonym for "there is an LMS in it".

### 5.3 Head B — predict the **acoustic state**, then map state → filter

The 2026 fixed-filter line has converged on prediction (PFANC predicts next frame's filter)
and on direction (D-SFANC selects by DoA; PD-SFANC predicts the DoA of a *moving* source with
a CRNN; SF-GFANC uses 3D spatial cues — distance, elevation, azimuth — jointly with frequency
cues in a multi-task CRNN). RHEAR takes the idea but restructures it into two stages:

```
   h(t), ω(t)  ──►  f_pred  ──►  â(t+1|t)   predicted acoustic state
                                    │        (incl. predicted DoA θ̂)
                                    ▼
                                   g(·)      state → filter map
                                    │
                                    ▼
                              w₀(t+1) to L0
```

```
â(t+1 | t) = f_pred( h(t), ω(t) )          predict the state, not the filter
α(t+1)     = g( â(t+1|t) )                 map state to basis weights
w₀(t+1)    = Σ_k α_k(t+1) · b_k( θ̂ )       bounded basis combination
```

**Why predict state rather than predict the filter directly** (PFANC/PD-SFANC regress or
classify the filter itself):
1. The state trajectory is **directly supervisable** — in the simulator we know the true
   source bearing, periodicity and level at `t+1`, so `f_pred` gets a dense training signal
   instead of only the downstream control error.
2. It **decouples prediction from control**. Re-measuring `Ŝ(z)`, changing the basis, or
   re-tuning the controller does not require retraining the predictor.
3. It is **inspectable**: a predicted bearing track on the demo dashboard is something a
   panel can watch being right or wrong. A predicted filter index is not.

#### 5.3.1 The basis, and why direction does not multiply its size

Do **not** regress 256 free taps, and do **not** store a separate filter bank per direction
(PD-SFANC uses a 10° grid — 36 sectors — which for us would mean 36× the filter memory).
Regress mixing weights over a fixed, pre-trained basis:

```
w₀(t+1) = Σ_{k=1}^{K} α_k(t) · b_k ,        K ≈ 8–16
```

where `{b_k}` are the subband components of one pre-trained broadband control filter,
decomposed by a **perfect-reconstruction filter bank** (this is precisely GFANC's trick — it
needs very little prior data). `α(t) = tanh(W_B h(t))` allows sign inversion per band.

Two properties fall out for free:
- **Bounded output.** Any `w₀` is a bounded combination of known-stable filters, so a
  mis-classification degrades attenuation instead of destabilising the loop. Critical for a
  safety-relevant system.
- **Prediction, not reaction.** Note the index: the head emits the filter for frame `t+1`
  from frames up to `t`. That is PFANC's contribution — it removes the one-frame tracking lag
  that hurts SFANC/GFANC on rapidly varying noise, and defence noise is the definition of
  rapidly varying. The high-order-Markov argument for using several past frames is why the
  encoder has an 8-frame context.

`w₀(t+1)` is handed to L0 as the *starting point*; FxNLMS then refines it at 48 kHz. This is
the hybrid SFANC-FxNLMS structure: **neural for speed of response, classical for asymptotic
optimality.** Neither alone is enough.

Direction is folded in **physically rather than combinatorially.** What changes at the two
reference mics as a source moves in azimuth is, to first order, the inter-mic **delay and
level** — that is what ITD and ILD *are*. So a direction sector is parameterised as a
fractional-delay plus gain correction applied to the shared band basis:

```
b_k(θ)  ≈  g(θ) · D_{τ(θ)} { b_k }         D = fractional-delay (Farrow/Lagrange) operator
```

Memory cost: the 8–16 shared basis filters plus a small `(θ → τ, g)` table, instead of
`K × 36` stored filters. This is the difference between 8 KB and 300 KB, i.e. between fitting
in a ₹3,000 MCU's flash and not.

Cross-fade `w₀` over ~5 ms when switching to avoid a click.

#### 5.3.2 Head rotation is the moving-source problem — and we get the cue for free

PD-SFANC solves moving sources in a room, predicting **one frame = 0.5 s ahead**. A headset
has the same problem for a different reason and on a much harder timescale: **the listener
rotates.** A head turn runs at 100–300 °/s, so PD-SFANC's 0.5 s frame is 50–150° of relative
bearing change — far too coarse to copy directly. Our L2 frame is 16 ms (≈ 2–5° of rotation),
which is why the rate was set there.

But a headset also has a cue a room-ANC system does not: **a ₹100 MEMS gyroscope on the
headband measures the angular rate directly.** Relative bearing can then be *extrapolated
analytically* rather than inferred from audio:

```
θ̂(t+1) = θ(t) − ω_z(t)·Δt + (source's own motion, estimated from audio)
```

Audio-based DoA only has to supply the slow absolute bearing and correct drift; the fast
component comes from the IMU with essentially zero latency and no SNR dependence. That
inverts the hard part of the problem, and it is a contribution this literature does not have
because room-ANC systems have no head to instrument.

Two further payoffs:
- **Head motion resolves front/back ambiguity.** Two mics on a head give a cone of confusion;
  rotating the head and observing which way the bearing estimate moves disambiguates it. This
  is how human localisation does it, and the IMU tells us the rotation exactly.
- **Left and right cups get different filters.** A source at 60° azimuth has different primary
  paths to each ear. Non-directional SFANC picks one filter; directional selection picks two.

#### 5.3.3 Direction changes the causality budget — the headset-specific reason this matters

This is the argument to lead with, and it is stronger than the room-ANC framing.

§1.1 of the research base: the admissible electrical delay is `τ_primary − τ_secondary`, and
`τ_primary` is **the acoustic travel time from reference mic to ear, which is a function of
incidence angle.** A source directly ahead of a front-mounted reference mic gives the largest
primary delay and the most generous budget; a source from behind can give a *negative* margin,
where the noise reaches the ear before the reference mic has even seen it and feedforward
control is fundamentally impossible in that band. This is precisely what the free-field
causality-versus-direction study reports.

So direction is not a nice-to-have feature bolted onto filter selection. It is an input to
three separate decisions:

| θ̂ feeds | effect |
|---|---|
| causality margin `Δ(θ)` | how much of the delay budget L0 may spend, per direction |
| achievable attenuation | bands where `Δ(θ) < 0` are declared uncontrollable — the controller **stops trying** there instead of injecting uncorrelated energy |
| filter/basis selection | delay-gain corrected basis, per ear |

That third row is a safety property, not just performance: a feedforward controller operating
outside its causal region does not merely fail to cancel, it *adds* noise. Knowing `θ` is how
you know when to stop.

#### 5.3.4 What we will not claim about direction

SF-GFANC estimates distance, elevation and azimuth in 3D. We will not, and saying so up front
is worth more than overclaiming:

- **Azimuth in the horizontal plane: yes**, at maybe 15–30° resolution from two reference mics
  spaced ~15 cm with a head between them (ITD below ~1.5 kHz, ILD from head shadow above it).
- **Front/back: only with head motion or a third non-collinear mic.** The boom mic helps.
- **Elevation: no.** Elevation cues are HRTF spectral notches, which a 2-mic shell array on a
  cheap headset cannot resolve. Adding it would require a 4-mic array and pinna-adjacent
  placement, which is out of budget.
- **Distance: no.** Head-sized arrays cannot triangulate range beyond ~1 m; the only usable
  proxy is direct-to-reverberant ratio, which is unreliable outdoors.
- **DoA degrades exactly when we need it.** Wind noise decorrelates the reference pair and
  PD-SFANC's own CRNN is reported at >90% DoA accuracy only for SNR ≥ 20 dB. So DoA
  confidence gates its use — this is what `κ` in head A is for, and low `κ` falls back to the
  omnidirectional broadband filter.

### 5.4 Head C — adaptation scheduler

Emits `(μ, λ_leak, p, mode)` per frame. Learned by policy distillation from an offline
grid-search oracle (which `(μ,λ,p)` maximised attenuation on this scene) — cheap, supervised,
no RL needed.

### 5.5 Head D — SE conditioning

`γ(t), θ(t) ∈ ℝ^32` FiLM parameters injected into L1's bottleneck:
`ĥ = γ ⊙ h_SE + θ`. This is how "the enhancer knows a helicopter is overhead" becomes a
concrete, differentiable mechanism instead of a slide bullet.

### 5.6 Head E — speech presence / hear-through gate

Per-band speech presence probability on the *external* field. Drives `TRANSPARENT` mode and
supplies the SPP auxiliary target for L1's multi-task loss.

### 5.7 Why two small networks (L1 + L2) and not one merged multi-task network

The tempting simplification is a single tiny network emitting speech mask + acoustic state +
control coefficients from one shared trunk. We are deliberately not doing that, for four
reasons — while still capturing most of the sharing benefit through the FiLM link:

1. **Different inputs.** L1 observes the boom mic (16 kHz, near-field speech). L2 observes the
   reference and error mics (outside field, no speech from the wearer). A merged trunk has to
   ingest both at all times.
2. **Different duty cycles.** L2 must run continuously — the controller needs coefficients
   whether or not anyone is speaking. L1 only matters when the wearer transmits. Separate
   networks let L1 be gated off entirely during listen-only operation, which is most of the
   time. Merged, you pay the union of both budgets always. Two nets at ~100 k + ~40 k params
   is *cheaper in average power* than one merged net, not more expensive.
3. **Different rates.** 250 Hz vs 62.5 Hz. Merging forces one of them to run at the other's
   rate, wasting 4× compute or losing 4× responsiveness.
4. **Failure isolation.** L2 feeds a safety-relevant control loop. A change to the speech
   model should not be able to alter the control filter. Keeping the coefficient generator in
   its own network with a bounded-basis output (§5.3) is a containment boundary worth having.

What we *do* share: the scene embedding `h(t)`, joint training, and one export/quantisation
pipeline. That is the multi-task benefit — a shared representation — without the coupling.

---

## 6. L1 — Communication path: streaming speech enhancement

Input: boom mic `z(n)` at 16 kHz. Output: `ŝ(n)` to the radio. This is the path the PS's
STOI/PESQ/SNR numbers are scored on.

### 6.1 Low-latency analysis/synthesis

```
frame 32 ms (512), hop 4 ms (64), asymmetric windows
algorithmic latency  L_alg = L_synth + hop = 4 ms + 4 ms = 8 ms
```

Asymmetric windows decouple latency from spectral resolution: a long analysis window keeps
32 ms of frequency resolution while only the short synthesis window enters the latency term,
subject to the COLA condition on the window *product*. Without this trick, 32 ms resolution
costs 32 ms of delay.

Optional Phase-3 upgrade for a sub-millisecond variant: predict the taps of a **minimum-phase
FIR** filter and apply it sample-by-sample — the published hearable route to 0.32–1.25 ms
algorithmic latency at 4.1 dB SI-SDRi.

### 6.2 Network — GTCRN-class, conditioned

```
       Z (complex STFT, 257 bins)
                 │
        power-law compression |Z|^0.3 ⊙ e^{jθ}
                 │
        ┌────────┴────────┐
        │ ERB/subband split (equivalent-rectangular-bandwidth grouping)
        └────────┬────────┘
                 │
     grouped depthwise-separable conv encoder  (5 stages, channel shuffle)
                 │
     ┌───────────┴────────────┐
     │  FiLM  ⟵ γ(t), θ(t) from L2 head D
     └───────────┬────────────┘
                 │
     dual-path recurrent core:
        · intra-frame (frequency) GRU  — captures harmonic structure
        · inter-frame (time)  GRU      — captures temporal continuity
        · temporal recurrent attention (TRA) gate
                 │
     grouped conv decoder  (skip connections)
                 │
        ┌────────┴────────┬─────────────┬──────────────┐
        │ mask head       │ SPP head    │ noise-class  │  (aux, training only)
        │ M = (M_r, M_i)  │ q̂(t,f)      │  head        │
        └────────┬────────┴─────────────┴──────────────┘
                 ▼
     complex-ratio-mask application (bounded, DCCRN-E form)
```

**Mask formulation** — decoupled magnitude and phase, magnitude bounded so the network
cannot invent energy:

```
|M| = tanh( |M_raw| )                       ∈ [0,1)
θ_M = atan2( M_i , M_r )
Ŝ(t,f) = |M|·|Z(t,f)| · exp( j·( θ_Z(t,f) + θ_M(t,f) ) )
```

Bounding `|M|` by `tanh` is not cosmetic: an unbounded mask under a 160 dB transient is a
divergence waiting to happen.

**Budget:** ≤ 100 k parameters, ≤ 60 MMAC/s — i.e. GTCRN (48.2 k / 33.0 MMAC/s) and LiSenNet
(37 k / 56 M MAC/s) class. Any architecture choice that breaks this budget is rejected
regardless of its PESQ.

**Conditional computation** (LiSenNet's idea): head E's speech-presence output gates whole
encoder stages during speech-absent frames. Typical duty cycle in a comms headset is low, so
this is a large average-power win for free.

### 6.3 Optional dual-channel mode

The reference mics see noise *without* speech (they are outside the cup, away from the mouth).
That is a nearly ideal noise-only observation — a luxury single-channel SE research does not
have. Feeding `x_L, x_R` as an auxiliary noise reference into the encoder is the H-GTCRN-style
dual-channel extension and should pay off most exactly where the PS is hardest: very low SNR.
Ship single-channel first; add this as measured ablation.

---

## 7. Loss functions

This section *is* axiom A3 made concrete.

### 7.1 The full objective

```
L = λ_sisdr · L_SI-SDR
  + λ_mag   · L_mag
  + λ_ri    · L_RI
  + λ_asym  · L_asym          ← intelligibility-first term
  + λ_spp   · L_SPP
  + λ_cls   · L_cls
```

**Scale-invariant SDR** (time domain, phase-sensitive by construction):

```
s_target = (⟨ŝ,s⟩ / ‖s‖²) · s ,   e_noise = ŝ − s_target
L_SI-SDR = −10·log₁₀( ‖s_target‖² / ‖e_noise‖² )
```

**Power-law-compressed spectral losses** (compression `c = 0.3` matches perceptual loudness
and stops loud frames dominating the gradient):

```
L_mag = E[ ( |S|^c − |Ŝ|^c )² ]
L_RI  = E[ | |S|^c e^{jθ_S} − |Ŝ|^c e^{jθ_Ŝ} |² ]
```

### 7.2 The asymmetric residual term (RHEAR-specific)

Split the spectral error into *over-suppression* (we removed speech) and *under-suppression*
(we left noise), and weight them differently:

```
Δ(t,f) = |S(t,f)|^c − |Ŝ(t,f)|^c

L_asym = β_over  · E[ max(0, +Δ)² ]      # speech was attenuated  → punished hard
       + β_under · E[ max(0, −Δ)² ]      # noise was retained     → punished gently

with   β_over / β_under  =  ρ(c_class) ∈ [3, 10]
```

`ρ` is **scheduled by the acoustic state from L2 head A** — primarily impulsiveness `I`,
predictability `H` and the external-speech flag, with the class posterior as a tiebreaker:

| scene | ρ | reasoning |
|---|---|---|
| rotor / engine (stationary, tonal) | 3 | noise is predictable; suppress harder safely |
| wind (broadband LF) | 4 | |
| babble / speech present | 8 | over-suppression here deletes the message |
| impulse / post-blast | 10 | maximum conservatism; do not destroy the recovered speech |

This is the operational encoding of the CAS residual-noise-control finding: aggressive noise
removal is not the same as better perceived speech. It also gives the ablation study a clean
knob — "here is PESQ/STOI vs ρ" is a graph that wins arguments.

### 7.3 Auxiliary heads

```
L_SPP = BCE( q̂(t,f), q(t,f) ),   q = 1[ |S(t,f)|² > η·|N(t,f)|² ]
L_cls = CE( p̂(c), c )
```

The SPP auxiliary target is a documented, essentially free improvement — the shared encoder
learns *where speech is* rather than only *what to output*, which is what protects weak
consonants. Weight the multi-task terms by learned homoscedastic uncertainty:

```
L_total = Σ_i ( 1/(2σ_i²) · L_i + log σ_i )
```

so the λ's are learned rather than hand-tuned, per the multi-task SPP paper.

### 7.4 What we deliberately do not do

No adversarial/GAN objective in v1. It buys MOS on paper, costs stability and training time,
and its failure mode is hallucinated speech — the single worst failure mode in a defence
communication system. Revisit only after the metric targets are met.

---

## 8. Training data engine

Generalisation to unseen defence noise is the actual deliverable. The data pipeline matters
more than the architecture, and it is the cheapest thing to get right.

### 8.1 Mixture generator

For each training example, sample:

```
y(n) = h_s * s(n)  +  Σ_{i=1}^{K} g_i · (h_i * n_i)(n)  +  a(n)  +  w(n)

s        ~ DNS clean speech (+ Indian-accented English/Hindi command corpus — record our own)
n_i      ~ MAD ∪ NOISEX-92 ∪ AudioSet mined,  K ~ U{1,3}
h_s, h_i ~ ISM-simulated RIRs (cup interior, vehicle cabin, open field, corridor)
g_i      : set so total SNR ~ U(−10, +15) dB      ← note the negative tail; this is defence
a(n)     ~ α-stable impulse train, α ~ U(1.2, 1.9), random onset/duration
w(n)     ~ mic self-noise
```

Then apply, with probability, the augmentations that mirror real hardware pathologies:

| Augmentation | Models |
|---|---|
| soft/hard clipping, `tanh(κ·y)` | ADC and preamp saturation under blast |
| μ-law/A-law codec round trip, packet loss | the actual radio link |
| band-limiting, LF roll-off | boom-mic response and wind screen |
| level drift, DC offset | cheap analog front end |
| **memoryless polynomial + FIR (Hammerstein)** on the anti-noise path | **driver/amp nonlinearity** — the documented cause of classical ANC instability |
| random mic gain mismatch ±3 dB | manufacturing tolerance across our 3 mics |
| time-varying SNR within one utterance | a vehicle driving past mid-sentence |

The last one deserves emphasis: **static-SNR training is the reason models fail on
non-stationary noise.** SNR must be a trajectory, not a scalar.

### 8.2 Curriculum

1. Stationary noise, SNR 0–15 dB → get the model to converge at all.
2. Add non-stationary + multi-source + SNR trajectories.
3. Add impulses and clipping (the hardest; introduce last).
4. Fine-tune on **our own rig recordings** — the domain gap between YouTube-sourced MAD audio
   and our cup's actual response is the gap that decides the live demo.

### 8.3 Held-out protocol (non-negotiable)

Noise *types* held out entirely, not just clips: train on engine/rotor/siren, test on tank
and machine-gun from NOISEX-92. Reporting in-distribution numbers only would be the easiest
way to lose credibility with a DRDO panel.

---

## 9. Budgets

### 9.1 Latency

**The L0 figures below are design targets, not universal requirements.** The admissible
electrical delay depends on reference-mic position, headset geometry, the secondary path,
noise incidence direction, the target attenuation bandwidth, and the predictability of the
noise (§5.2). We therefore commit to the *procedure*, not the number:

> For the targeted headset geometry and broadband ANC bandwidth, we measure the causality
> budget experimentally (swept-sine primary/secondary path measurement + measured loopback
> delay) and design L0 to fit inside the measured budget with margin, reporting that margin
> as a live quantity.

The numbers below are what we expect that measurement to allow, and what we design against
until it is done.

| Segment | Budget | Notes |
|---|---|---|
| **L0 ANC loop** | | |
| ADC group delay | ≤ 30 µs | needs a low-latency codec; sigma-delta decimation filters are the usual trap |
| FIR + update | ≤ 25 µs | 256 taps @ 48 kHz on an M7 |
| DAC + amp | ≤ 40 µs | |
| **L0 total** | **≤ 100 µs (design target)** | against an *estimated* τ_primary ≈ 120–290 µs for this geometry; to be replaced by the measured budget |
| **L1 comms path** | | |
| algorithmic (asym. STFT) | 8 ms | |
| inference | ≤ 3 ms | RTF ≈ 0.75 at worst on target MCU |
| I/O + buffering | ≤ 4 ms | |
| **L1 total** | **≤ 15 ms** | inside the 20 ms comms target |
| **Hear-through** | ≤ 2 ms end-to-end | analog-assisted or minimum-phase FIR; > 2 ms causes comb filtering |

### 9.2 Compute

| Block | MMAC/s | Where |
|---|---|---|
| L0 FIR ×2 ch + updates | ~50 | MCU core / DSP |
| L1 speech enhancer | ≤ 60 | NPU or MCU |
| L2 scene engine | ≤ 15 | MCU, 62.5 Hz duty |
| **Total** | **≤ 125 MMAC/s** | comfortably inside a 300–600 MHz Cortex-M7/M55-class part |

For calibration: GTCRN is 33.0 MMAC/s at 48.2 k params; LiSenNet 56 M MAC/s at 37 k params;
the sub-ms hearable model needed 376 MIPS. We are asking for less than a phone's always-on
audio budget.

### 9.3 Model size

| | params | INT8 size |
|---|---|---|
| L1 enhancer | ≤ 100 k | ≤ 100 KB |
| L2 scene engine | ≤ 40 k | ≤ 40 KB |
| filter basis `{b_k}` | 16 × 256 | 8 KB (INT16) |
| direction table `(θ → τ, g)` | 24 sectors × 2 | < 1 KB |
| **Total** | | **< 200 KB** — fits internal flash |

Note what the delay-gain parameterisation of direction (§5.3.1) bought: a 10°-grid filter
bank in the PD-SFANC style would have added ~300 KB and pushed us off the part.

---

## 10. Metrics and acceptance criteria

The PS's targets, disambiguated and ranked. **Rank order is the contribution; do not
reorder it in the pitch.**

| # | Metric | Definition | Target | PS ref |
|---|---|---|---|---|
| 1 | **STOI / ESTOI** | intrusive, on `ŝ` vs `s`, 16 kHz | **≥ 0.85** at 0 dB input SNR | ✓ |
| 2 | **Word/phrase accuracy** | human panel on a defence phrase set (call signs, digits, grid refs) | ≥ 95 % at 0 dB | our addition |
| 3 | **PESQ (WB)** | ITU-T P.862.2 | **≥ 2.5** | ✓ |
| 4 | **ΔSI-SDR** | `SI-SDR(ŝ) − SI-SDR(z)` | **≥ 15 dB** at 0 dB input | disambiguates "SNR > 15 dB" |
| 5 | Speech distortion / over-suppression rate | `E[max(0,Δ)²]` on speech-active bins | report, minimise | our addition |
| 6 | **ANC NMSE at error mic** | `10log₁₀(Σe²/Σd²)` | ≤ −10 dB, 100 Hz–1 kHz | protection loop |
| 7 | Impulse peak reduction | peak SPL at ear, passive+limiter | report per ANSI S12.42 method | safety |
| 8 | **Latency** | measured loopback, not calculated | L0 ≤ 100 µs, L1 ≤ 20 ms | ✓ |
| 9 | RTF, MMAC/s, model size, power | measured on target | ≤ 0.75, ≤ 125 MMAC/s, < 200 KB | ✓ embedded |

Metric 4 matters: "SNR > 15 dB" is ambiguous between output SNR, segmental SNR and
improvement. State the definition before the panel asks, and report all three.

---

## 11. Deployment path

```
PyTorch (float32, streaming-aware from day 1 — no offline-only ops)
   │  · causal convs only, no future context anywhere
   │  · explicit GRU state in/out, exported as ONNX inputs/outputs
   ▼
ONNX (opset with fixed shapes, one frame in / one frame out)
   │
   ├── laptop demo:  onnxruntime, float32/INT8
   ├── edge board :  TensorRT (Jetson) — matches the PS's named platform
   └── MCU        :  post-training quantisation, mixed FP16/INT8
                     (published to work for RNN-based SE on multi-core MCUs)
   ▼
INT8 model + measured latency/power on hardware
```

Quantisation notes that will actually bite: GRU state is the first thing to break under INT8
— keep recurrent state in FP16 (this is exactly the mixed FP16/INT8 result), quantise conv
weights per-channel, and calibrate on *low-SNR* data, because activation ranges under a
120 dB scene are nothing like those on VoiceBank-DEMAND.

**Streaming-aware from day 1** is the single most-skipped step. A model trained with
non-causal ops or full-utterance normalisation cannot be made streaming later without
retraining. Build the streaming wrapper in week 1 and test parity (offline vs frame-by-frame
output must match to < 1e-4) at every checkpoint.

---

## 12. Known failure modes and mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| ANC howl / instability | phase error in `Ŝ` > 90°, or fit change | online `Ŝ` tracking, leakage, band-limited control, hard `‖w‖` cap |
| Divergence during gunfire | α-stable, `E[e²]` undefined | log-LMS score function **+** detector freeze (both) |
| Speech deleted as noise | over-suppression | asymmetric loss, SPP head, ρ scheduling |
| Model fine offline, poor live | domain gap; non-streaming training | rig fine-tuning; streaming parity test in CI |
| Great on MAD, fails on tank | noise-type leakage across splits | hold out noise *types*, not clips |
| Clipped input | ADC saturation on blast | analog limiting before ADC, high-AOP mics |
| Classifier wrong → bad filter | misclassification | bounded basis combination: worst case is *less* attenuation, never instability |
| USB demo "ANC" doesn't work | causality violated by 25–100× | do broadband ANC on the MCU; laptop does SE + periodic-noise ANC only |

---

## 13. What each experiment proves

Design the evaluation so every claim maps to one measurement.

| Claim | Experiment |
|---|---|
| The model helps | ΔSTOI/ΔPESQ/ΔSI-SDR vs unprocessed and vs a Wiener/spectral-subtraction baseline, on held-out noise types |
| Better than classical ANC | NMSE of RHEAR L0 vs plain FxLMS on the same rig, same noise |
| Neural filter selection is what helps | ablation: fixed filter / SFANC-select / GFANC-generate / PFANC-predict, all with the same FxNLMS refinement |
| It survives impulses | inject SαS + real gunshot; plot `‖w‖` and NMSE across the event for LMS vs log-LMS vs log-LMS+freeze |
| Intelligibility-first works | PESQ/STOI vs ρ sweep; human phrase-accuracy panel |
| Scene conditioning helps | ablation with FiLM conditioning removed |
| Direction helps | attenuation vs source azimuth, with and without directional basis selection; show the band where `Δ(θ) < 0` and that the controller correctly disengages |
| Head rotation is tracked | wearer rotates at ~150 °/s under a fixed source: NRL over time for (a) fixed filter, (b) audio-only DoA, (c) IMU-fused prediction |
| It is real-time | measured loopback latency + RTF + MMAC/s on target, not calculated |
| It fits a headset | INT8 model size, measured MCU inference time, measured power |

The ablation row 3 is the paper. Everything else is the product.

---

## 14. Open questions for the team

1. **Codec choice** drives the whole L0 budget — need a part with published ADC/DAC group
   delay in µs, not "low latency" marketing. Candidate class: TDM audio codecs with a
   low-latency/decimation-bypass mode.
2. **Is the boom mic single or dual?** A second boom mic enables a fixed differential
   beamformer that costs almost nothing and kills a lot of diffuse noise before the NN.
3. **Speech corpus.** Indian-accented military phraseology does not exist as an open dataset.
   Recording ~2 h of team-read call signs / grid references / digit strings is a half-day of
   work and is probably worth more than any architectural improvement.
4. Do we target **ESTOI** in addition to STOI? ESTOI correlates better under modulated noise,
   which is what rotor noise is.
