# RHEAR — Digital Twin
### Prove it on the laptop, then buy
v0.1 · 2026-08-27 · code in `rhear/`, run with `./run_experiments.sh`

---

## 0. What this answers

*"Can we test all of this digitally, with near-enough accuracy and some proof, before
spending money?"*

**Mostly yes.** The control architecture, the latency budget, the part selection, the
robustness claims and the model's metrics are all provable in simulation. The acoustics of a
real cup on a real head are not. So the twin's job is not to replace the rig — it is to
**decide which hardware is worth buying, and to have every algorithm already working when it
arrives.**

This is also standard practice in the field rather than a shortcut: PD-SFANC's results are
numerical simulations, Deep ANC trains and evaluates on simulated acoustic paths, and the
speech-enhancement literature is almost entirely offline. What is *not* standard, and what
this document insists on, is validating the simulator against known answers before trusting
any number it produces.

---

## 1. Validating the simulator first (E00)

A twin that cannot reproduce results we already know is worthless. Two checks run before
anything else, and both must pass.

### V1 — Derive the codec latency instead of quoting it

The hardware doc claims a generic 48 kHz audio codec has 500–600 µs of decimation group
delay. Rather than assert it, the twin *builds* a typical sigma-delta decimation chain and
measures it:

```
CIC order 4, R=16                           9.8 us
halfband FIR 47 taps @ 192 kHz            119.8 us
halfband FIR 95 taps @ 96 kHz             489.6 us
TOTAL                                     619.1 us     <- derived, not asserted
literature for 48 kHz audio codecs        500-600 us
```

**619 µs against a literature range of 500–600 µs.** Independent derivation, same
order of magnitude, same conclusion. The remaining spread is filter sharpness, which is a
design choice, not a modelling error.

### V2 — Obey the information bound

Achievable ANC attenuation cannot beat `NR ≤ −10·log₁₀(1 − γ²)`. The twin solves the
Wiener optimum at four coherence levels:

| γ² | theoretical bound | achieved | gap |
|---|---|---|---|
| 0.500 | −3.01 dB | −3.07 dB | −0.06 |
| 0.900 | −10.00 dB | −10.00 dB | 0.00 |
| 0.990 | −20.00 dB | −20.01 dB | −0.01 |
| 0.999 | −30.00 dB | −30.00 dB | 0.00 |

**Matches theory to 0.06 dB across four orders of magnitude of coherence.**

> V2 failed on its first run, by 3–4 dB *in our favour* — always the suspicious direction.
> Cause: the uncorrelated component was white while the signal was band-limited, so γ² was
> frequency-dependent and a single scalar bound was the wrong comparison. Fixed by shaping
> the uncorrelated part to match. **This is what validation is for**, and it is why no
> number in §2 or §3 should be believed without §1.

---

## 2. E01 — The codec decision, proved

Geometry: reference mic 7 cm from the ear, driver 2 cm.

```
tau_primary   = 204.1 us
tau_secondary =  58.3 us
CAUSALITY BUDGET for the whole electrical chain = 145.8 us
```

Three noise characters spanning the predictability axis, with `H` measured at the budget
delay (this is head A's `predictability` descriptor, computed for real):

| signal | H at 146 µs |
|---|---|
| harmonic (200 Hz + 8 harmonics, rotor-like) | **0.9841** |
| band-limited broadband (80–2000 Hz, vehicle-like) | **0.9120** |
| wideband (200–8000 Hz, wind/blast-tail-like) | **0.0303** |

### Attenuation (dB, more negative is better) vs total electrical delay

| τ_e | harmonic | band-limited | **wideband** |
|---|---|---|---|
| 0 µs | −59.6 | −52.7 | **−14.1** |
| 5 µs | −48.9 | −52.6 | **−13.8** |
| 100 µs | −49.4 | −45.2 | **−13.2** |
| **146 µs** (budget) | −49.6 | −39.2 | **−9.0** |
| 200 µs | −42.4 | −38.7 | **−4.4** |
| 300 µs | −36.3 | −37.6 | **−0.7** |
| 619 µs | −18.1 | −35.1 | **−0.0** |
| 1000 µs | −24.1 | −30.8 | **−0.1** |

### Read as parts

| part | delay | harmonic | band-limited | wideband |
|---|---|---|---|---|
| analog ANC IC (AS3415 class) | 0 µs | −59.6 dB | −52.7 dB | **−14.1 dB** |
| **ADAU1777/1787 ANC codec** | 5 µs | −48.9 dB | −52.6 dB | **−13.8 dB** |
| generic 48 kHz codec | 619 µs | −18.1 dB | −35.1 dB | **−0.0 dB** |
| laptop over USB audio | ~6 ms | — | — | **~0 dB** |

**The hardware argument is now a measurement.** On genuinely unpredictable noise the ANC
codec retains 13.8 of the 14.1 dB available, and the generic codec delivers *nothing* — the
loop is a no-op. That is a part-selection decision worth ₹1,500, decided for free.

### Four findings that came out of it

**F1 — The cliff sits where theory says.** Wideband degradation begins right at the 146 µs
budget (−13.2 dB at 100 µs → −9.0 at 146 → −4.4 at 200 → −0.7 at 300).

**F2 — Achieved attenuation tracks the prediction floor.** At 200 µs the theoretical
Δ-step prediction floor is −3.5 dB and the loop achieves −4.4 dB; at 619 µs the floor is
−0.1 dB and the loop achieves −0.0 dB. The twin **explains** the cliff rather than displaying
it, which is the difference between a simulation and a demonstration.

**F3 — Predictability, not bandwidth or level, sets the latency budget.** Three signals at
identical level, ordered by `H`, ordered identically by delay tolerance. This is direct
evidence for the architecture's decision (§5.2) to regress `H` rather than a class label —
and it is why staging the hardware plan around periodic noise first is legitimate physics,
not a dodge.

**F4 — Exploiting periodicity costs filter length.** The harmonic prediction floor as a
function of predictor order, at 619 µs delay (one period = 960 samples):

| order | 256 | 512 | **1024** | 2048 | 4096 |
|---|---|---|---|---|---|
| floor | −2.2 dB | −4.4 dB | **−17.4 dB** | −19.0 dB | −19.5 dB |

The floor collapses the moment the filter exceeds **one period**. The ANC run with L=1600
achieved −18.1 dB, sitting exactly between the 1024- and 2048-tap floors.

> **Design rule this generates:** beating the causality constraint on periodic noise requires
> a control filter at least one fundamental period long. For 20 Hz rotor blade-pass at
> 192 kHz that is **9,600 taps** — a real memory and MAC cost, and a concrete argument for
> running low-frequency periodic control in a **decimated subband** rather than at the full
> rate. This did not come from the literature; it came from the twin.

**F5 (secondary) — Usable step size shrinks with loop delay.** The largest stable µ fell from
0.02 at 0 µs to 3×10⁻⁴ at 300 µs. Latency charges twice: once through causality, once through
convergence rate.

---

## 3. E02 — Impulsive noise, and a claim that did not survive

Physically-grounded setup: background at 95 dB SPL, α-stable burst (α = 1.5) peaking at
**160 dB SPL**, reference mic AOP **135 dB SPL** (the Infineon IM73A135 figure), passive cup
25 dB. The event exceeds the mic's AOP by 25 dB, so the measured peak is compressed by 25 dB
— the controller sees a **saturated reference against an unsaturated error**.

| controller | pre-event | during | post | recovery |
|---|---|---|---|---|
| **FxLMS (unnormalised)** | **diverged** | diverged | diverged | never |
| FxNLMS | −9.4 dB | −1.5 dB | −10.1 dB | 50 ms |
| FxNLMS + log score | −6.9 dB | −1.1 dB | −6.9 dB | 50 ms |
| + detector freeze | −6.9 dB | −1.1 dB | −6.9 dB | 50 ms |

### Two corrections to our own design

**Correction 1 — normalisation is itself an impulse defence.** The first version of this
experiment failed to reproduce any divergence at all. Reason: NLMS divides the update by
`‖x̂‖²`, which divides out a loud reference. The textbook divergence result applies to
**unnormalised** FxLMS, which does diverge here, unrecoverably. Our documents implied FxNLMS
was at risk. It is not, and saying so precisely is stronger than the vaguer claim.

**Correction 2 — the adaptation freeze does not earn its place.** Ablation across step sizes
and burst counts, measuring post-event attenuation:

| scenario | µ | log | log + freeze | Δ |
|---|---|---|---|---|
| single burst | 0.01 | −6.51 | −6.36 | +0.15 |
| single burst | 0.05 | +4.12 | +3.89 | −0.22 |
| single burst | 0.20 | +15.87 | +15.85 | −0.02 |
| burst train (8) | 0.01 | −6.28 | −6.51 | −0.23 |
| burst train (8) | 0.05 | +4.15 | +4.13 | −0.02 |
| burst train (8) | 0.20 | +15.92 | +16.07 | +0.14 |

**Largest effect anywhere: 0.23 dB.** Normalisation plus the log score function already carry
it. We proposed the freeze as an adaptation-stability mechanism and simulation says that
specific claim is unsupported.

The detector is **not** deleted — it still earns its place for the analog limiter trigger,
`PROTECT` mode, hear-through ducking and event logging, all of which are about the ear rather
than the filter. But `02-architecture.md` §4.4 has been corrected: the belt-and-braces framing
was wrong, and the braces were doing all the work.

> Note the positive dB at µ = 0.05 and 0.2 — the controller **adding** energy. Same finding
> as E01/F5, arrived at independently: too much step size for the loop delay is its own
> failure mode, separate from impulsiveness.

---

## 3a. E06 — Does neural filter selection actually beat the alternatives?

This is the project's central claim, tested. Non-stationary scenario, 6 s, five noise changes:
`engine → rotor → wind → engine → siren → wind`. Reference-mic self-noise at 45 dB SNR, so
attenuation is capped near the coherence bound rather than by deterministic-signal artefacts.
Electrical delay set to **38 µs**, the ADAU1772 figure — i.e. the part we actually chose.

Four controllers on identical signals:

| controller | overall | **250 ms after a change** | steady state |
|---|---|---|---|
| fixed generic filter, no adaptation | −7.02 dB | −10.99 dB | −12.38 dB |
| FxNLMS only (the classical baseline) | −8.11 dB | −11.39 dB | −29.67 dB |
| **learned selection + FxNLMS refine** | **−12.19 dB** | **−20.67 dB** | −31.95 dB |
| oracle selection + FxNLMS refine (ceiling) | −13.43 dB | −23.37 dB | −30.16 dB |

**+9.3 dB over plain FxNLMS in the 250 ms after a noise change**, closing **77 %** of the gap
to an oracle that knows the noise class perfectly. +4.1 dB overall.

The win lands exactly where the architecture predicted it would: **at the transitions.** In
steady state FxNLMS catches up on its own (−29.7 dB), which is precisely why the design pairs
neural selection with classical refinement instead of choosing one — selection buys the
transient, adaptation buys the asymptote. Neither alone is enough, and now that is a
measurement rather than a design opinion.

The selector is **1,764 parameters**, reaching 83.2 % frame accuracy on the scenario (100 % on
its training distribution — the gap is frames that straddle a transition, which is honest and
expected). It runs at 62.5 Hz and emits a filter choice, never audio.

### The twin caught its own design rule being violated

First run of E06 used `L = 256` and the **engine filter failed to train — +0.22 dB**, i.e.
worse than doing nothing. Cause: the engine fundamental is 50 Hz, one period is 960 samples
at 48 kHz, and E01/F4 says exploiting periodicity needs a filter spanning at least one period.
Raising `L` to 1024 fixed it immediately (−53 dB on its own noise). A rule derived in one
experiment predicted a failure in another. That is the twin being internally consistent, and
it is the strongest evidence so far that it is modelling physics rather than curve-fitting.

---

## 3b. E05 — Direction and head rotation: **G5 PASSES**

Rebuilt from scratch as a **true two-cup headset**. The previous version rendered one ear
and scored the other cup's reference against it; **no ANC number from it is quoted anywhere.**

This version has independent `d_L` and `d_R`, all four reference→ear pairings, an independent
L0 controller and filter state per ear, and left/right/joint scoring.

### The geometry, unit-tested (`tests/test_geometry.py`)

Every margin is checked against a closed form derived by hand, not against the
implementation: **292 cases, |error| < 1e-12.** Plus mirror symmetry, the straight-ahead
identity `F/c − τ_s`, and "never causal from behind".

The result that reframed the experiment: a headset has **four** feedforward pairings, not two.

```
              L->L      R->L      L->R      R->R
  -90 deg    -14.6    -612.2    +495.6    -102.0     us
    0 deg    +87.5     +87.5     +87.5     +87.5
  +90 deg   -102.0    +495.6    -612.2     -14.6
```

For a source at +90° the left cup's **own** reference is 102 µs non-causal while the **right**
reference leads the left ear by ~496 µs. Allowing contralateral pairings takes each ear from
**38% → 68% of azimuth** with a usable reference.

### Bearing estimation, unit-tested (`tests/test_doa.py`)

| test | result |
|---|---|
| known angles, broadband | **0.30° RMSE** |
| known angles, harmonic + 20% broadband | 5.12° RMSE |
| pure line spectrum | **46° @16 ms, 27.7° @500 ms — a bias, not noise** |
| coherence high / coherent-spread low on a clean tone | 0.97 / 0.0065 |
| confidence predicts error | **corr −0.820** |
| gate 0.4 separates | accepted 3.2° vs rejected 30.5° |
| independent channels | 0.000 |

**A two-microphone ITD estimator is fundamentally ambiguous for a line spectrum**, and more
data does not help. Adding 20% broadband content fixes it. Real engine and rotor noise carries
broadband content, so this is a caveat rather than a blocker — but it must be stated.

### The confidence metric, replaced (requirement 3)

Peak-to-sidelobe margin is **wrong** for this application: a tonal source has a periodic
correlation surface, so the margin collapses even when the bearing is perfect. The replacement
is `coherence × f(coherent spread)`:

- **coherence** — are the channels linearly related? High for a clean tone.
- **coherent spread** — flatness of `coh(f)·P(f)`. Low for a line spectrum, where the bearing
  genuinely *is* wrong by ~30°.

Only the product is a useful gate. An intermediate version used the spread of a *single*
channel and was fooled exactly backwards: independent noise raised flatness without making the
delay resolvable, so **confidence rose from 0.24 to 0.84 as SNR fell** while the error stayed
at 30°. Coherence weighting removes the independent noise from the estimate.

> The first draft of the test asserted "confidence must be high for a clean tonal source".
> That was wrong — for a clean tonal source the bearing is bad, so confidence there should be
> **low**. The only property worth testing is that confidence *predicts error*.

### Requirement 8 — the hypothesis, tested as a 2×2 factorial (oracle bearing)

| config | joint attenuation | vs A |
|---|---|---|
| **A** fixed ipsilateral ref + omni filter | −3.47 dB | — |
| **B** reference SELECTION + omni filter | −4.12 dB | **−0.66 dB** |
| **C** fixed ref + per-sector FILTER | −4.06 dB | **−0.59 dB** |
| **D** both | −5.90 dB | **−2.44 dB** |

> **Hypothesis** — *"the primary value of direction is causal reference selection, not
> directional filter-shape optimisation"* — **NOT SUPPORTED.**

Neither mechanism dominates. They are almost equal alone (−0.66 vs −0.59 dB) and strongly
**super-additive** together: −2.44 dB against −1.25 dB for the sum of the parts. The reason is
structural — the filter bank is trained per *(reference, sector)* pair, so a sector filter is
only correct once the matching reference is selected, and a selected reference is only fully
exploited with its matching filter. **They are one mechanism, not two.**

The corrected conclusion for the architecture: keep directional selection, but specify it as
*joint* reference-and-filter selection. Implementing either half alone recovers about a quarter
of the benefit.

### Requirement 8, isolated: where the effect actually lives

Static bearings, config A vs D, no rotation:

| bearing | ipsi margin | best ref | A | D | gain |
|---|---|---|---|---|---|
| −30° | +89.8 µs | L | −3.75 | −4.46 | −0.72 dB |
| 0° | +87.5 µs | L | −3.98 | −4.71 | −0.73 dB |
| +30° | +46.1 µs | R | −3.84 | −6.75 | **−2.90 dB** |
| **+60°** | **−23.3 µs** | R | −3.65 | −13.01 | **−9.36 dB** |
| **+90°** | **−102.0 µs** | R | −3.71 | −13.21 | **−9.51 dB** |
| +120° | −169.1 µs | R | −3.71 | −7.20 | −3.49 dB |

**The entire effect lives where the ipsilateral reference goes non-causal.** Inside ±50° it is
worth 0.7 dB; at +60…+90° it is worth **9.4 dB**.

> This also caught a scope error of mine. The first run swept ±78°, which is almost entirely
> inside the ipsilateral causal region, and measured only −0.47 dB. The sweep must cross the
> causal boundary or the experiment cannot see what it is testing. Now ±110°.

### A claim of ours that this overturned

At −90°/−120° **no** reference is causal. The architecture (§5.3.3) said a controller outside
its causal region *adds* energy and must disengage. Measured: disengaging scores **0.00 dB**
while continuing to run scores **−3.5 dB** — because the source is partly periodic, and
periodic components are cancellable despite the delay (E01/F3 again).

**Corrected rule: disengage only when the noise is both non-causal *and* unpredictable.** That
is head A's `predictability` descriptor doing real work, and the experiment now degrades
gracefully to the ipsilateral reference instead of muting.

### Requirements 5–7 — the ablation

Baseline is config A, **no direction at all**. (An earlier run used "direction machinery driven
by a constant bearing of zero" as the baseline, which is much stronger and hid a 2.4 dB effect
behind a 0.9 dB one.)

| bearing source | DoA RMSE | tracking lag | ref-sel accuracy | fallback | joint att | after a turn |
|---|---|---|---|---|---|---|
| fixed (no direction) | — | — | 43–50 % | — | −3.47 dB | −3.58 dB |
| audio only | 11.1–11.3° | 0–128 ms | 99–100 % | 0 % | **−5.94 dB** | −6.47 dB |
| IMU dead-reckoning | 0.6–2.1° | 0–112 ms | 99–100 % | 0 % | −5.89 dB | −6.23 dB |
| audio + IMU fusion | 4.4–6.5° | 0–128 ms | 99–100 % | 0–1 % | −5.90 dB | −6.63 dB |
| oracle | 0° | — | 100 % | — | −5.93 dB | −6.07 dB |

Usable-reference availability: **86–88 % of frames per ear** across the ±110° sweep.
Rotation rate (30 / 100 / 250 °/s) changes attenuation by < 0.4 dB — the mechanism is not
rate-limited in this range.

**Gyro drift — why fusion exists.** A 3 s run cannot show it (0.5 °/s bias = 1.5°), so it was
measured over increasing durations:

| duration | IMU | audio | fusion |
|---|---|---|---|
| 3 s | **0.9°** | 12.0° | 6.4° |
| 12 s | 3.3° | 10.1° | 6.6° |
| 30 s | 8.7° | 10.9° | **7.7°** |

The IMU is best briefly and degrades without bound; audio is noisy but drift-free; **fusion is
the only one that does not degrade with time**, and it wins by 30 s. This is the complementary
behaviour the architecture assumed, now demonstrated rather than asserted.

> Finding this required fixing a **double-import bug** — the drift loop set `DUR` on a second
> copy of the module (`import rhear.experiments.e05_direction` from inside `__main__`), so
> `run()` kept reading 3.0 s and the table showed a gyro that never drifted.

**Computational cost.** GCC-PHAT 21.1 + coherence 7.7 = **28.8 MMAC/s**, against the
architecture's 15 MMAC/s L2 budget — **that budget is wrong and must be raised or the DoA rate
cut.** Gyro integration is 0.0002 MMAC/s. Measured wall time 0.39–0.45 ms/frame, 2.4–2.8 % of
the 16 ms frame.

### Requirement 12 — decision

> ## G5: **PASS**
> A realisable bearing source improves ANC by **2.42–2.47 dB** over the no-direction baseline,
> capturing essentially all of the oracle's 2.46 dB. The criterion is the ANC gain, not the
> bearing accuracy — and note that **audio-only has the worst bearing RMSE (11.3°) yet the best
> attenuation (−5.94 dB)**, which is exactly why the gate was specified this way.

Keep the directional mechanism, specified as **joint reference-and-filter selection**. Use
fusion rather than either source alone: it costs the same as audio and is the only bearing
source that is stable over minutes.

---

## 4. What the twin can and cannot prove

| Claim | Provable in sim? | Status |
|---|---|---|
| Causality budget and the delay cliff | **Yes** | E01 ✓ |
| Codec selection (5 µs vs 619 µs vs USB) | **Yes** | E01 ✓ |
| Predictability governs delay tolerance | **Yes** | E01 ✓ |
| Filter length needed for periodic noise | **Yes** | E01/F4 ✓ |
| Robust update under impulses | **Yes** | E02 ✓ |
| Mic AOP saturation behaviour | **Yes** (given a datasheet AOP) | E02 ✓ |
| STOI / PESQ / SI-SDR of the enhancer | **Yes** — this is how the whole field works | E03, not built yet |
| Model size, MACs, INT8 accuracy loss | **Yes**, exactly | E04, not built yet |
| Bearing estimation and its degradation | **Yes** | **E05 ✓ (S1)** |
| Direction converting into ANC gain | **Yes** | **E05 ✓ — +2.42 dB, see §3b** |
| L2 filter generation beating a fixed filter | **Yes** | **E06 ✓ — +9.3 dB after a change** |
| MCU inference time, power | **Projection only** | needs hardware |
| **Passive attenuation / NRR of a real cup** | **No** | needs hardware |
| **Real secondary path, driver nonlinearity, fit variation** | **No** | needs hardware |
| **Real codec group delay vs datasheet** | **No** | needs hardware |
| **Sim-to-real gap itself** | **No, by definition** | needs hardware |

**The honest boundary:** simulation is strong for the *control* problem, because it is linear
systems plus known nonlinearities. It is weak for *passive acoustics and fit*, which is where
most of the dB actually comes from in a real headset. So the twin decides the electronics and
the algorithms; the rig decides the acoustics.

---

## 5. The buy gate

Do not order the integrated prototype until every one of these passes in simulation:

- [x] **G0** Simulator validated (V1 codec delay, V2 coherence bound) — **PASS**
- [x] **G1** Causality cliff located, and part selection decided by measurement — **PASS**
- [x] **G2** Impulse robustness demonstrated; unsupported claims removed — **PASS**
- [ ] **G3** L1 enhancer hits STOI ≥ 0.85, PESQ ≥ 2.5, ΔSI-SDR ≥ 15 dB at 0 dB SNR on
      held-out noise *types* (E03)
- [ ] **G4** INT8 quantised model ≤ 200 KB, ≤ 125 MMAC/s, with measured accuracy loss (E04)
- [x] **G5** Directional selection beats a fixed filter during head rotation (E05) —
      **PASS**: +2.42 dB over a no-direction baseline from a realisable bearing source,
      capturing all of the oracle's 2.46 dB. Rebuilt as a true two-cup headset with
      unit-tested geometry and a new coherence-based confidence metric
- [x] **G6** L2 filter generation beats a fixed filter on non-stationary noise (E06) —
      **PASS**: +9.3 dB after a change, 77 % of the oracle gap closed, 1,764-param selector

G3 and G4 need datasets (DNS, MAD, NOISEX-92); G5 needs none. All are still **no hardware**.
Only after all six does money get spent — and by then the algorithms are written, so the rig's job narrows to
measuring the two things simulation genuinely cannot: the cup and the secondary path.

**One purchase is worth making before the gate:** the ADAU1777 sample, because its lead time
is the risk, not its cost.

---

## 6. Running it

```bash
./run_experiments.sh
```

No datasets, no downloads, no hardware. numpy + scipy only. Total runtime ≈ 3 minutes on a
laptop. Results land in `results/*.npz`.

| file | what it is |
|---|---|
| `rhear/sim/electronics.py` | virtual mics, ADC/DAC decimation, AOP saturation, part catalogue |
| `rhear/sim/paths.py` | primary/secondary acoustic paths, fractional delays, head geometry |
| `rhear/core/anc.py` | FxLMS/FxNLMS with robust score functions, impulse detector |
| `rhear/core/predict.py` | Δ-step prediction floor — the theory the results are checked against |
| `rhear/core/signals.py` | broadband, harmonic, α-stable, gunshot-like bursts |
| `rhear/core/l2runtime.py` | the frame-rate **coefficient interface** — slow selector writes `w`, fast loop runs FxNLMS per sample |
| `rhear/core/state.py` | acoustic state features (periodicity, impulsiveness, flatness, centroid) |
| `rhear/sim/geometry.py` | head geometry, per-cup causal cones, time-varying source rendering |
| `rhear/core/doa.py` | banded GCC-PHAT, gyro model, complementary-filter bearing tracker |
| `rhear/experiments/` | E00 validation, E01 causality, E02 impulsive, E06 filter selection |

The virtual patch panel is `CODECS` in `electronics.py`: swapping ADAU1777 for a generic
codec is one dictionary lookup, which is what makes "connect the electronics virtually"
literal rather than a metaphor.
