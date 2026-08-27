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
| Direction / head-rotation tracking | **Yes** (spherical-head model in `paths`) | E05, not built yet |
| L2 filter generation beating a fixed filter | **Yes** | E06, not built yet |
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
- [ ] **G5** Directional selection beats a fixed filter during simulated head rotation (E05)
- [ ] **G6** L2 filter generation beats a fixed filter on non-stationary noise (E06)

G3–G6 need datasets (DNS, MAD, NOISEX-92) but still **no hardware**. Only after all six does
money get spent — and by then the algorithms are written, so the rig's job narrows to
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
| `rhear/experiments/` | E00 validation, E01 causality, E02 impulsive |

The virtual patch panel is `CODECS` in `electronics.py`: swapping ADAU1777 for a generic
codec is one dictionary lookup, which is what makes "connect the electronics virtually"
literal rather than a metaphor.
