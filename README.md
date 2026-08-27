# RHEAR
**Robust Hearing Enhancement & Active Reduction** — an AI/ML-enabled adaptive noise
cancellation and speech-enhancement system for defence communication headsets.

> SIH PS **26052** · DRDO · Dept. of Defence Production / iDEX · Category: Hardware

---

## The one-paragraph version

Defence headsets face two different problems that get wrongly treated as one. **Protecting
the ear** from engine, rotor, wind and blast noise is an acoustics and control problem bound
by microsecond-scale causality. **Protecting the message** — keeping "fire", "hold", a grid
reference intelligible over the radio — is a statistical estimation problem bound by
perceptual latency, and it is what the PS's STOI/PESQ/SNR targets actually measure. RHEAR
builds both as one system with **three loops at three rates**, and keeps neural computation
out of the ultra-low-latency ANC control loop — there the model emits *filter coefficients*,
not audio — while a separate small network processes the communication path directly, where
the budget is milliseconds rather than microseconds. That split is what makes a ₹15,000
headset running a 33 MMAC/s model a credible answer to "real-time performance on embedded
hardware".

## Core architecture in one picture

```
L0   48 kHz     protection + ANC control     pure DSP, ≤ 100 µs      → the ear
L1   250 Hz     streaming speech enhancer    ≤ 100 k params, ≤ 15 ms → the radio
L2   62.5 Hz    noise scene engine           feeds coefficients + conditioning to both
```

**Neural computation stays out of the ANC control loop.** L2 speaks to L0 in *coefficients* —
on the order of 16 numbers, 62.5 times a second — never audio. On the communication path a
small network *does* process audio directly, and that is fine: its deadline is ~15 ms, not
~100 µs, and an INT8 speech enhancer of this class has already been deployed on-chip inside a
headphone. The distinction is the latency budget of the loop, not a blanket rule about AI.

## Targets

| | target | status |
|---|---|---|
| STOI | ≥ 0.85 @ 0 dB SNR, held-out noise types | not started |
| PESQ (WB) | ≥ 2.5 | not started |
| ΔSI-SDR | ≥ 15 dB | not started |
| ANC attenuation @ error mic | ≤ −10 dB, 100 Hz–1 kHz | not started |
| L0 loop latency | inside the *measured* causality budget for our geometry (design target ≈ 100 µs) | not started |
| L1 end-to-end latency | ≤ 20 ms | not started |
| Model footprint | < 200 KB INT8, ≤ 125 MMAC/s | not started |
| Headset BOM | ₹4,200–6,100 at qty 1k (→ ₹12–18 k field price) | costed, parts not ordered |

## Documents

| | |
|---|---|
| [`docs/01-research-base.md`](docs/01-research-base.md) | Literature review, the four physical constraints, verified numbers, corrections to earlier design notes |
| [`docs/02-architecture.md`](docs/02-architecture.md) | Full architecture: signal model, control math, network design, loss functions, data engine, budgets, metrics |
| [`docs/03-roadmap-and-hardware.md`](docs/03-roadmap-and-hardware.md) | Build A / Build B BOMs, demo design, 12-week plan with gates, risk register |
| [`docs/04-hardware-design.md`](docs/04-hardware-design.md) | Part-level design: codec selection (the decision that makes or breaks it), 5-mic plan, MCU, sample rates, memory, power/battery arithmetic, volume BOM, buildability verdict |
| [`docs/05-digital-twin.md`](docs/05-digital-twin.md) | **Runnable simulation.** Validated twin, causality/codec proved by measurement, impulse robustness, and the buy gate that must pass before money is spent |

## Three things we will not claim

1. **That AI cancels a gunshot.** Gunshot peaks run 144–172 dB SPL; hearing protection is
   passive attenuation plus analog limiting. The AI restores intelligibility through and
   after the event, and keeps the controller from diverging. That claim is defensible; the
   other one is not.
2. **That broadband ANC runs over USB from a laptop.** USB audio round-trip is 5–20 ms
   against a causality budget on the order of a few hundred µs. The fast loop runs on a
   microcontroller at the mics. The exact budget is geometry-dependent and we measure it
   rather than quote it.
3. **That more dB is better.** A system that suppresses 20 dB and drops a word has failed.
   The loss function, the metrics and the demo all rank intelligibility first.

## How to describe RHEAR in one sentence

A low-cost hybrid headset in which a lightweight neural model estimates the acoustic
environment and the optimal control parameters, a deterministic high-rate adaptive controller
performs the actual cancellation, and a separate low-latency neural path preserves
communication intelligibility.

Mapped onto the PS's hard requirements:

| PS requirement | RHEAR answer |
|---|---|
| stationary noise | adaptive FxNLMS over pre-trained filters |
| non-stationary noise | predictive neural filter generation (PFANC-class), with IMU-fused bearing prediction for head rotation |
| impulsive noise | detector-gated adaptation freeze + robust update + analog limiting + passive |
| speech intelligibility | separate neural communication path, intelligibility-first loss |
| embedded hardware | L0 in a 5 µs ANC codec, L1+L2 ≈ 90 MMAC/s on an NPU MCU, < 200 KB INT8, no GPU |
| ₹10–20 k | budget spent on acoustics; compute is already nearly free |

## Status

Research base, architecture and part-level hardware design complete. **Digital twin running
and validated** — the causality/codec decision and the impulse-robustness claims are now
measurements rather than assertions (`docs/05-digital-twin.md`). No hardware purchased.

```bash
./run_experiments.sh     # ~3 min, numpy + scipy only, no datasets, no hardware
```

Buy gate: G0–G2 **pass**; G3–G6 (speech-enhancement metrics, quantisation, direction, filter
generation) still to build — all of them simulation-only, still no hardware needed.
Next: E03, the L1 enhancer against STOI/PESQ/SI-SDR targets.
