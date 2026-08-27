# RHEAR — Hardware, Build Plan and Roadmap

Companion to `02-architecture.md`. Prices are Indian market estimates and **need verification
before ordering** — treat every ₹ figure here as a planning number, not a quote.

> **Superseded in part by [`04-hardware-design.md`](04-hardware-design.md)**, which does the
> part-level selection (codec, mics, MCU, power) and the volume BOM. Where the two differ,
> `04` is current — in particular, the L0 control filter now lives in an ANC codec or an
> analog filter network, not on the MCU, and the target BOM is a qty-1k number.

---

## 1. Two builds, deliberately

| | **Build A — Dev Rig** | **Build B — Target Headset** |
|---|---|---|
| Purpose | prove algorithms, collect data, live demo | prove the ₹10–20 k product thesis |
| Compute | laptop (L1, L2) + MCU (L0) | single MCU/DSP/AI-SoC (L0+L1+L2) |
| Cost | ₹6,000–8,500 | ₹12,000–16,000 BOM |
| When | weeks 1–6 | weeks 7–12 |

Build A is not a throwaway: its mic geometry, `Ŝ(z)` measurement and recordings all transfer.

---

## 2. Build A — development rig

| Item | Qty | Est. ₹ | Why exactly this |
|---|---|---|---|
| Closed-back **passive** headphone (no ANC) | 1 | 1,200–2,000 | Must be passive. A commercial ANC can is a black box and invites "what did *you* build?" |
| Electret/MEMS mic capsules + preamps (outer ref ×2, in-cup error ×1) | 3 | 600–1,200 | The error mic is what makes attenuation *measurable* rather than claimed |
| Boom/headset mic | 1 | 400–800 | Communication path — the STOI/PESQ path |
| 4-channel USB audio interface | 1 | 1,500–2,500 | 4 simultaneous inputs is the real requirement; 2-ch will not do |
| **MCU board** (STM32H7 class) for L1/L2 + coefficient link | 1 | 1,800–3,000 | No hard µs deadline once L0 moves off it |
| **Analog fast-loop board** (op-amps + RC network + CD4053 analog switches, 4 selectable responses) | 1 | 300–600 | The L0 controller. Zero converter group delay; selective-fixed-filter ANC in analog. See `04` §8 |
| ADAU1777 sample + breakout (parallel track, not on the critical path) | 1 | 1,000–2,000 | Production fast loop; order week 1 for lead time |
| **MEMS IMU (gyro) on the headband** | 1 | 150–400 | Head angular rate → analytic DoA extrapolation (arch §5.3.2). Cheapest high-value part in the build |
| Cables, mounts, foam, 3M tape, breadboard | — | 400–800 | |
| **Total** | | **≈ ₹7,100–11,800** | trims to ~₹6 k if the MCU board is borrowed |
| Reference monitor speaker for noise playback | 1 | *borrow* | Needed to reproduce engine/rotor/blast at calibrated SPL |
| SPL meter (or calibrated phone app + one-point calibration) | 1 | 500–1,500 | Numbers on the demo dashboard have to come from somewhere |

**Mic geometry is an algorithmic decision** (coherence bound, research base §1.2). Mount the
outer pair so the reference-to-ear acoustic delay is *maximised* subject to still being
coherent with what reaches the ear — front-facing on the cup shell, not on the headband.
Measure `γ²_xd(f)` before committing to a position; the measured curve is the upper bound on
everything L0 can achieve and it belongs on a slide.

### What runs where in Build A

```
ref L, ref R, error  ──►  MCU (48 kHz)  ──►  L0: FIR + robust FxNLMS + limiter ──► driver
       │                     ▲
       │                     │ w₀, μ, mode over UART/USB @ 62.5 Hz
       │                     │
       └──► USB interface ──► LAPTOP: L2 scene engine  +  L1 speech enhancer ──► "radio" out
```

The coefficient link is low-bandwidth (16 × float per 16 ms ≈ 4 kB/s), so a plain UART is
enough. This link *is* axiom A1 made physical, and it is the thing to point at when a judge
asks how a laptop model can run in a headset: **the neural output crossing that wire is
16 numbers every 16 ms, not audio.**

---

## 3. Build B — target headset BOM (₹15 k centre case)

| Component | Est. ₹ | Notes |
|---|---|---|
| Rugged closed-back cups + acoustic seal + headband | 3,000–4,000 | Target NRR ≥ 24 dB. This layer does more work than the AI |
| Reference MEMS mics ×2 (high AOP ≥ 130 dB) | 700–1,200 | AOP rating is the blast-survival spec |
| Error mic ×1 (in-cup) | 250–500 | |
| Boom mic ×1 (+ optional second for differential) | 500–1,000 | |
| Low-latency multi-channel audio codec | 1,000–1,800 | **Select on published group delay in µs.** Highest-risk part |
| MCU / DSP / AI-SoC (Cortex-M7/M55 + NPU class) | 2,500–4,000 | Must hold ≤ 125 MMAC/s, < 200 KB model, ~µs interrupt latency |
| Speaker drivers ×2 | 800–1,500 | Need clean LF output — ANC is an LF technology |
| Analog limiter / compressor front end | 300–600 | Before the ADC. Cheap, and it is the actual hearing protection |
| MEMS IMU (gyro) | 150–400 | Head rotation cue for directional filter prediction |
| Battery + charging + PMIC | 800–1,200 | |
| PCB + connectors + assembly | 1,000–1,800 | |
| PTT interface | 400–800 | |
| **Total** | **≈ ₹11,250–18,400** | ₹15 k realistic centre |

**Cost decisions that matter:** extra money goes into *acoustics* (seal, drivers, mic AOP,
codec latency), never into a bigger network. The network is already free at 33 MMAC/s. The
seal is not.

---

## 4. The demo

What a judge sees, in the order they see it:

```
┌───────────────────────────────────────────────────────────────────────┐
│  RHEAR LIVE            ROTOR · P 0.91 · H 0.87 · I 0.04 · θ̂ 042°→038° │
├───────────────────────────────────────────────────────────────────────┤
│  reference mic      91.4 dB SPL   ████████████████████                │
│  error mic (ear)    72.1 dB SPL   ███████████                         │
│  ── attenuation     19.3 dB       (100 Hz – 1 kHz, measured)          │
├───────────────────────────────────────────────────────────────────────┤
│  COMMS PATH                     bypass        RHEAR                   │
│  STOI                            0.61          0.88                   │
│  PESQ (WB)                       1.42          2.71                   │
│  SI-SDR                         −0.4 dB       15.8 dB                 │
├───────────────────────────────────────────────────────────────────────┤
│  L0 loop latency      86 µs   │ causality margin  +134 µs        ✓    │
│  L1 total latency   13.7 ms   │ RTF 0.61                              │
│  model 178 KB INT8  │ 118 MMAC/s │ MCU load 41 %                      │
└───────────────────────────────────────────────────────────────────────┘
```

*(Layout only — every number above is a placeholder until measured.)*

Three things this dashboard does that a demo video cannot:
1. **Causality margin as a live number, against a budget we measured on this headset** —
   nobody else at the hackathon will be making this claim, and it is the one that actually
   proves embedded feasibility.
2. **A/B on the comms path** with the panel wearing the headset and a team-mate speaking
   over rotor noise at 90 dB. Let them press bypass themselves.
3. **The scene label changing** as you switch the noise source. It makes the AI legible.

Demo matrix to fill in with real measurements:

| Noise | passive only | +ANC (L0) | +AI (L1) | full RHEAR |
|---|---|---|---|---|
| Vehicle engine | | | | |
| Helicopter rotor | | | | |
| Wind | | | | |
| Siren | | | | |
| Impulse (recorded gunshot, safe SPL) | | | | |
| Babble | | | | |

Columns are cumulative — that is what makes the architecture's layering visible.

---

## 5. Roadmap (12 weeks, week-relative)

### Phase 0 — Foundations (weeks 1–2)
- [ ] Repo, environment, data download (DNS, MAD, NOISEX-92), licence audit
- [ ] **Mixture generator** with SNR trajectories, α-stable impulses, RIRs, clipping (§8, arch)
- [ ] Metric harness: STOI/ESTOI, PESQ-WB, SI-SDR, and the over/under-suppression split
- [ ] **Streaming wrapper + parity test in CI from day 1** (offline vs frame-by-frame < 1e-4)
- [ ] Order Build A parts

**Gate:** a Wiener-filter baseline scored end-to-end on held-out noise types.

### Phase 1 — Communication path (weeks 3–5)
- [ ] GTCRN-class L1 model, causal, asymmetric-window STFT, 8 ms algorithmic latency
- [ ] Full loss (SI-SDR + compressed mag/RI + asymmetric + SPP multi-task)
- [ ] Hit **STOI ≥ 0.85, PESQ ≥ 2.5, ΔSI-SDR ≥ 15 dB** at 0 dB on held-out noise types
- [ ] ρ sweep (asymmetric-loss ratio) → the intelligibility-vs-suppression curve

**Gate:** PS metric targets met offline. Nothing else proceeds until this is true.

### Phase 2 — Rig and protection loop (weeks 4–7, overlaps)
- [ ] Assemble Build A; measure `γ²_xd(f)` for 3 mic positions; pick the best
- [ ] Measure `Ŝ(z)` (swept sine); check phase error < 90° over the control band
- [ ] **Establish the measured causality budget**: swept-sine primary/secondary path
      measurement for this geometry + measured electrical loopback delay → the admissible
      `τ_electrical`, per noise incidence direction. This number is geometry-specific and
      replaces every estimate in the architecture doc
- [ ] Classical FxNLMS on the MCU, designed to fit inside that measured budget with margin
- [ ] Robust update + impulse detector; reproduce LMS divergence under SαS, then fix it
- [ ] Record our own corpus: 3-mic + boom, real noise playback, Indian-accented phraseology

**Gate:** ≥ 10 dB measured attenuation, 100 Hz–1 kHz, on the rig, together with the measured
causality budget and the margin we operate inside it.

### Phase 3 — Integration (weeks 7–9)
- [ ] L2 scene engine: **acoustic state head** (periodicity, impulsiveness, predictability,
      bearing, confidence — class posterior as auxiliary), filter-basis decomposition,
      state-prediction → filter map (PFANC/PD-SFANC-style, at our 16 ms frame not their 0.5 s)
- [ ] **Direction**: measure `Δ(θ) = τ_primary(θ) − τ_secondary` across azimuth on the rig;
      identify the angles/bands where feedforward control is non-causal and make the
      controller disengage there
- [ ] **IMU fusion**: gyro on the headband, analytic bearing extrapolation, front/back
      disambiguation by head motion; ablate against audio-only DoA under head rotation
- [ ] Coefficient link MCU ↔ host; cross-fade; bounded-basis safety check
- [ ] FiLM conditioning of L1 from L2
- [ ] Full ablation: fixed / select / generate / predict, each ± FxNLMS refinement
- [ ] Fine-tune L1 on rig recordings

**Gate:** the ablation table is filled in; neural filter generation shows a measurable win
over a fixed filter on non-stationary noise; and directional selection shows a measurable win
during head rotation.

### Phase 4 — Embedded and demo (weeks 9–12)
- [ ] ONNX export → INT8 (recurrent state FP16), calibrated on low-SNR data
- [ ] Measure on target: inference time, MMAC/s, memory, power
- [ ] Jetson path as well, since the PS names it explicitly
- [ ] Live dashboard; demo matrix filled with measurements; human phrase-accuracy panel
- [ ] Documentation: architecture, results, honest limitations, Build B BOM and path

**Gate:** the dashboard runs live for 10 minutes with someone else wearing the headset.

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Audio codec group delay blows the L0 budget | **High** | High | Select the part on published µs specs in week 1; fall back to analog-assisted feedback ANC |
| Broadband ANC gives < 10 dB on a leaky cheap cup | Medium | Medium | Improve seal first; report band-limited attenuation honestly; hybrid FF+FB |
| MCU cannot hold L1 in real time | Medium | High | Model is already GTCRN-class; conditional computation; fall back to Jetson for the demo and report MCU projection with measured MACs |
| MAD's YouTube audio doesn't transfer to our rig | Medium | Medium | Rig fine-tuning is already in Phase 3; keep a held-out rig test set |
| Impulse handling never gets tested at realistic SPL | High | Low-Med | Test with SαS injection + recorded gunshot at safe replay level; be explicit that full-SPL validation needs a shock tube / ANSI S12.42 lab |
| DoA from 2 mics too coarse/unstable to be useful | Medium | Low | IMU carries the fast component; `κ` gates audio DoA; fallback is the omnidirectional filter, so the failure mode is "no directional gain", not "worse than baseline" |
| Team spends weeks on model architecture instead of data | **High** | High | Phase gates. The data engine ships in week 2, before any model tuning |

---

## 7. Team split (suggested)

| Track | Owns | Deliverable |
|---|---|---|
| **Data + metrics** | mixture generator, corpora, metric harness, CI parity test | the number everyone else is judged on |
| **Model (L1)** | enhancer, losses, ablations, quantisation | STOI/PESQ/SI-SDR targets |
| **DSP/embedded (L0)** | MCU, codec, FxNLMS, `Ŝ(z)`, latency measurement | causality margin + attenuation dB |
| **Scene engine (L2)** | classifier, filter generator, conditioning | the ablation table |
| **Hardware + demo** | rig build, mic geometry, dashboard, pitch | what the panel actually experiences |

Data + metrics is the highest-leverage seat, not the model seat. Staff it accordingly.
