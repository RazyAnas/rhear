# RHEAR — Hardware Design
### Is this actually buildable at ₹10–20 k? Part-level answer.
v0.1 · 2026-08-27 · companion to `02-architecture.md`

Tags: `[v]` verified against a datasheet/distributor this session · `[e]` engineering estimate ·
`[?]` must be confirmed before ordering. USD→₹ at ≈ ₹88/$.

---

## 0. Verdict first

**Yes, at volume. No, as a one-off prototype — and that distinction is the honest answer.**

| | qty 1 (prototype) | qty 1,000 (product BOM) |
|---|---|---|
| Electronics | ₹6,000–9,000 | ₹1,900–2,600 |
| Acoustics + mechanical | ₹4,000–6,000 | ₹1,800–2,800 |
| PCB + assembly | ₹2,000–4,000 (amortised over a 5-board run) | ₹450–700 |
| **Total** | **≈ ₹14,000–20,000** | **≈ ₹4,200–6,100** |

A ₹4,200–6,100 BOM supports a ₹12,000–18,000 field price at normal defence-electronics
margins. So the ₹10–20 k target is a **volume-BOM target that closes**, and anyone quoting a
qty-1 prototype cost as the product price is comparing the wrong numbers.

Three findings drove the design, in order of importance:

1. **A generic audio codec cannot do feedforward ANC.** Decimation-filter group delay at
   48 kHz is **500–600 µs** `[v]` — five times the entire L0 budget. This is not a firmware
   problem and no MCU choice fixes it.
2. **Purpose-built ANC codecs solve it outright.** ADAU1777 / ADAU1787 give an analog-in to
   analog-out path **as low as 5 µs** `[v]`, with 4 ADCs and 2 DACs — exactly our mic count —
   at **~11 mW for typical stereo ANC** `[v]`.
3. **Which means the MCU has no hard real-time deadline at all.** Once L0 lives in the codec,
   the MCU only runs L1+L2 at frame rate. That converts the hardest constraint in the project
   into a part-selection decision, and frees us to pick the MCU on price and TOPS/W.

---

## 1. The part that decides the project: the codec

### 1.1 Why the obvious choice fails

Cheap I²S codecs (PCM1808, TLV320AIC3204, WM8731 class) use linear-phase sigma-delta
decimation filters. Reported group delay for such filters is **500–600 µs at 48 kHz** `[v]`,
because delay ≈ `(filter_length − 1) / (2·f_out)`. Against a primary-path delay of
120–290 µs (§1.1, research base), the margin is **negative before any processing happens**.

This is the quantitative form of the correction made earlier — and it now has a number
attached, which is much better than "USB is slow".

### 1.2 What real ANC silicon does instead

Production ANC parts run the control path at **192–768 kHz** and keep the converter path
short and minimum-phase; one documented approach samples at 3.072 MHz and runs ANC processing
at 192 kHz `[v]`. At 768 kHz a single sample is 1.3 µs, and the decimation filter shrinks with it.

### 1.3 Candidates

| Tier | Part | Latency (analog in → analog out) | Flexibility | Chip cost | Risk |
|---|---|---|---|---|---|
| **A — analog** | ams **AS3415** (feedforward) / **AS3435** (feedback+hybrid) / AS3418 (TWS) `[v]` | ≈ 0 (no converters in the loop) | filter shape set by external RC; a handful of responses via analog switches | ₹300–900 `[e]` | Low electrical risk, low flexibility. Integrated speaker driver, 35 mW stereo out, >110 dB SNR, 900 nV input-referred noise, integrated bypass `[v]` |
| **B — ANC codec** ★ | Analog Devices **ADAU1777**, 4 ADC / 2 DAC `[v]` | **as little as 5 µs** `[v]` | fully programmable filters; coefficients writable at runtime | ~₹800–1,800 `[e]` | Package `[?]` — sibling ADAU1772 is LFCSP `[v]`, which is hand-reflowable |
| **C — ANC codec + DSP** | **ADAU1787**, 4 ADC / 2 DAC + FastDSP (768 kHz) + 28-bit SigmaDSP (50 MIPS) `[v]` | **5 µs at fS = 768 kHz**, FastDSP bypass `[v]` | most capable: on-chip adaptive filtering possible | chip ≈ $22–24 ≈ **₹2,000** `[v]`; **eval board $603.90 ≈ ₹53,000** `[v]` | **WLCSP** (`BCBZ`) — not hand-solderable, needs fine-pitch assembly |
| **D — generic codec** | PCM1808 / TLV320 class | **500–600 µs** `[v]` | full | ₹150–400 | **Rejected for L0.** Usable only for periodic noise, where prediction covers the delay |

**Choice: B (ADAU1777) for the target build; A as the hackathon fast-loop and permanent
fallback; C only if the WLCSP assembly risk is retired; D explicitly excluded from L0.**

Note the trap in C: the chip is ₹2,000 but the eval board is ₹53,000. Buying the eval board
would consume three whole prototype budgets. It is a lab instrument, not a route to a product.

### 1.4 The mapping that makes this elegant

The ADAU1777/1787 FastDSP path is a **biquad cascade**, not an arbitrary 256-tap FIR. That
is not a limitation here — it is a match:

```
SFANC/GFANC "bank of pre-trained fixed control filters"
        ≡
sets of biquad coefficients living in the codec's fast path

L2 (62.5 Hz)  ──writes coefficients over I²C/SPI──►  codec biquads (5 µs path)
FxNLMS refinement (48 kHz, on MCU or SigmaDSP)  ──►  same coefficients, slower
```

So the architecture's L0/L2 split maps **one-to-one onto the silicon**: the neural side writes
coefficients at 62.5 Hz; the coefficients execute at 768 kHz with 5 µs of delay. The
filter-basis representation in §5.3 should therefore be re-parameterised in **biquad
coefficient space** rather than FIR taps for the production build. FIR taps remain the
representation for the rig and for simulation.

---

## 2. Microphones: five, and the analog/digital split matters

### 2.1 Count

| # | Position | Role | Type |
|---|---|---|---|
| 1 | Outer shell, left | reference L | **analog** MEMS, high AOP |
| 2 | Outer shell, right | reference R | **analog** MEMS, high AOP |
| 3 | In-cup, left | error L | **analog** MEMS |
| 4 | In-cup, right | error R | **analog** MEMS |
| 5 | Boom | speech | digital PDM or analog — latency-insensitive |

**Five, not three.** The cups are acoustically independent, so hybrid feedforward+feedback
control needs a *per-ear* error mic; and two outer references are what give azimuth (§5.3).
Mics 1–4 consume exactly the four analog ADC inputs on the ADAU1777/1787. Mic 5 goes to the
codec's digital mic input or straight to the MCU — its deadline is 15 ms, not 5 µs.

### 2.2 Analog, not PDM, for mics 1–4

PDM digital mics contain an ASIC whose clock-cycle latency **dominates group delay in the
mid-band** `[v]`. Feedforward ANC is exactly where that delay is unaffordable, and the
literature is explicit that analog feedforward is preferred for its smaller delay `[v]`.
Modern designs that do use PDM must add low-latency decimation circuitry to claw the delay
back `[v]` — cost and complexity we can avoid by choosing analog parts.

The boom mic is the exception and should be digital if convenient: it saves an ADC channel
and its latency is irrelevant.

### 2.3 Acoustic overload point — the blast-survival spec

- High-performance MEMS mics need **AOP 130–135 dB SPL**, where AOP is defined as the SPL at
  which THD exceeds 10 % `[v]`.
- **Infineon IM73A135** (XENSIV analog MEMS): **73 dB SNR, 135 dB SPL AOP, IP57** dust/water
  rated at the microphone via sealed dual-membrane construction `[v]`. This is the reference
  part for mics 1–2: analog (low latency), highest available AOP (blast), and ingress-rated
  (field use). All three requirements in one part.

**Honest limit:** 135 dB AOP still saturates on a 160–172 dB gunshot at close range. There is
no MEMS mic that does not. The response is architectural, not component-level: an attenuating
acoustic port on the outer mics to shift the blast into the linear range, analog limiting
before the ADC, and the detector-driven adaptation freeze (§4.2/§4.4) so a saturated reference
never gets to rewrite the control filter. Passive attenuation protects the ear regardless of
what the electronics do.

---

## 3. Compute: what is left after the codec takes L0

With L0 in the codec, the MCU's job is L1 (≤ 60 MMAC/s) + L2 (≤ 15 MMAC/s) + the FxNLMS
coefficient update ≈ **80–95 MMAC/s, with no microsecond deadline**. Candidates:

| Part | Compute | Notes | Unit price |
|---|---|---|---|
| **STM32N6** (Cortex-M55 + Neural-ART NPU) ★ | NPU-class INT8 | Best price/performance; ST tooling and Indian availability are the practical advantages | **$9–15 ≈ ₹800–1,350** `[v]` |
| **Ambiq Apollo510** (Cortex-M55) | 30× power-efficiency claim for AI/ML `[v]` | Best choice if battery life becomes the binding constraint | **$17 ≈ ₹1,500** `[v]` |
| **Alif Ensemble E1C** (M55 + Ethos-U55) | **46 GOPS** in a tiny footprint `[v]` | Smallest package; unit pricing not public `[?]` | `[?]` |
| **ListenAI CSK6012** | ARM Star + HiFi4 DSP + NPU, 300 MHz, **128 GOPS**, and an **integrated 4-in / 2-out audio codec** `[v]` | **This is the single-chip answer** — and it is the exact chip the INT8 GTFCRN headphone deployment ran on `[v]`. Would collapse codec + MCU into one part | `[?]` — Chinese sourcing and Chinese-language documentation are real risks for an Indian defence build |
| **STM32H7** (M7 @ 480 MHz) | CMSIS-NN, no NPU; a 1-core M7 @ 480 MHz with 1 MB SRAM handles 80 KB quantised models `[v]` | The safe, available, well-documented **prototype** choice | ₹700–1,400 `[e]` |

**Choice: STM32H7 for the rig (availability and documentation), STM32N6 for the target build
(price and NPU).** Keep CSK6012 in the report as the volume single-chip path — it is a
credible cost-down and it is directly evidenced by the GTFCRN deployment.

Watch the memory-power interaction: external memory activity has been measured at **40–45 mW,
roughly 50 % of total power** on STM32 tinyML deployments `[v]`. Fitting weights in internal
SRAM is therefore a *power* decision, not only a cost one — which is another reason the model
budget is < 200 KB.

---

## 4. Sample-rate plan

```
768 kHz   codec FastDSP path      L0 control filter          5 µs analog→analog
 48 kHz   I²S/TDM to MCU          FxNLMS update, L2 features 16 ms frames
 16 kHz   decimated boom mic      L1 speech enhancer         4 ms hop, 8 ms algorithmic
```

Three rates, each chosen by its deadline. Nothing runs faster than its loop requires, which is
where the power budget comes from.

---

## 5. Memory budget

| | size |
|---|---|
| L1 enhancer weights (INT8) | ≤ 100 KB |
| L2 scene engine weights (INT8) | ≤ 40 KB |
| Filter basis + direction table | ≤ 9 KB |
| STFT buffers, GRU state (FP16), ring buffers | ≈ 80–120 KB `[e]` |
| **Working set** | **≈ 230–270 KB** |

Fits internal SRAM on STM32H7 (1 MB) and STM32N6 (4.2 MB) with room to spare — so no external
PSRAM, so no 40–45 mW memory penalty.

---

## 6. Power and battery

| Block | Power |
|---|---|
| ANC codec, stereo ANC | **11.1 mW** `[v]` |
| 5 microphones | ≈ 3–5 mW `[e]` |
| MCU running L1+L2 (duty-cycled; L1 gated off when not transmitting) | 80–150 mW `[e]` |
| Speaker drivers (anti-noise + comms audio, average) | 20–50 mW `[e]` |
| PMIC losses, LEDs, PTT | 10–20 mW `[e]` |
| **Total average** | **≈ 125–235 mW** |

```
8 h endurance:   0.2 W × 8 h = 1.6 Wh  →  1.6 Wh / 3.7 V ≈ 430 mAh
12 h endurance:  0.2 W × 12 h = 2.4 Wh →  ≈ 650 mAh
```

A **600–800 mAh LiPo** (≈ 20–25 g, ₹200–400) gives 10–14 h. **Battery is a non-issue** — it
is the smallest line in the BOM and the least risky. That is worth saying explicitly, because
"AI in a headset" invites the assumption that power is the blocker. The codec doing L0 at
11 mW is why it isn't.

---

## 7. The four cost tiers

### Tier 0 — ₹0–1,500 · software only
Existing headset + laptop. Proves L1 speech enhancement live (STOI/PESQ/SI-SDR), proves
nothing about ANC. Start here in week 1 because it needs no procurement.

### Tier 1 — ₹7,000–11,000 · development rig
Per `03-roadmap-and-hardware.md`, plus the fast-loop decision below.

### Tier 2 — ₹14,000–20,000 · qty-1 integrated prototype
Custom PCB, ADAU1777 + STM32N6, 5 mics, battery, boom/PTT. **Above the ₹15 k headline** purely
because of qty-1 component pricing and a 5-board PCB run — this is normal and should be stated
rather than hidden.

### Tier 3 — ₹4,200–6,100 · volume BOM (1,000 units)

| Line | ₹ (qty 1k) |
|---|---|
| ANC codec (ADAU1777 class) | 300–450 |
| MCU (STM32N6 class) | 800–950 |
| 4 × analog MEMS mic, high AOP | 200–350 |
| 1 × boom mic | 60–100 |
| 2 × speaker driver | 180–300 |
| Battery 600–800 mAh + PMIC/charger | 220–330 |
| PCB, 4-layer, small, + passives | 180–280 |
| **Electronics subtotal** | **1,940–2,760** |
| Cups, headband, acoustic seal (moulded) | 1,200–2,000 |
| Boom arm, cable, PTT, connector | 400–700 |
| Assembly + test | 450–700 |
| **Total BOM** | **≈ 4,000–6,200** |

**This table is the commercial argument.** It shows the ₹10–20 k field price is reachable with
margin, and it shows where the money goes: **acoustics and mechanical are ~45 % of the BOM,
compute is ~20 %.** Which is the thesis from `02-architecture.md` (axiom A4) arriving as a
spreadsheet: buy seal and drivers, not TOPS.

---

## 8. Buildability verdict, and what to do in week 1

The one genuine schedule risk is that **a custom ADAU1777 board is a 4–6 week loop** (design,
fab, assembly, bring-up) and it is not a hackathon-timeline activity. So:

**Hackathon fast loop: build it analog.** A hand-built hybrid feedforward+feedback analog ANC
filter — op-amps, an RC network, and CD4053-class analog switches to select among 4 response
shapes under MCU control — costs ~₹300–600, has **zero converter group delay**, and is a
literal analog implementation of selective-fixed-filter ANC. L2 selects the response over two
GPIOs at 62.5 Hz. Four states covers `ANC_TONAL`, `ANC_BROAD`, `TRANSPARENT`, `PROTECT`.

That is not a compromise dressed up as a feature. It is the correct hackathon embodiment: it
demonstrates the exact architecture (neural selection, deterministic execution) with parts
available locally and no PCB spin, and it is unambiguously ours in a way that reusing a
commercial ANC module would not be.

Order in week 1:
1. Headphone shell, 5 mics (2 × IM73A135 class for the outer pair), op-amps, analog switches,
   4-ch USB interface, MCU dev board, IMU, LiPo — the Tier 1 rig.
2. **One ADAU1777 sample plus a breakout**, in parallel, to start the package/footprint
   question (`[?]`) moving immediately. Its lead time, not its cost, is the risk.

---

## 9. Hardware risk register

| Risk | L | I | Mitigation |
|---|---|---|---|
| ADAU1777 package turns out to be WLCSP too | Med | High | AS3415/AS3435 analog tier is a complete fallback; CSK6012 is a second |
| Import lead time / customs on ADI + Infineon parts | **High** | Med | Order week 1; analog tier uses locally-available jellybean parts |
| Custom PCB does not come back in time | **High** | Med | Analog fast loop needs no PCB spin; Tier 2 is explicitly post-hackathon |
| Hand-built analog filter is noisy / drifts | Med | Low | Low-noise op-amps; the error mic *measures* the result, so degradation is visible, not hidden |
| Cheap cup leaks; passive attenuation under spec | Med | **High** | Measure NRR early; passive is ~45 % of the BOM for a reason. Budget to upgrade the seal |
| CSK6012 documentation is Chinese-only | High | Low | Report-only path; not on the critical route |
| 135 dB AOP mic saturates on real gunfire | **Certain** | Med | Attenuating port + analog limiter + adaptation freeze; never claim otherwise |

---

## 10. Open items — confirm before ordering

1. `[?]` **ADAU1777 package** (LFCSP vs WLCSP) and qty-1 India-landed price. Single highest-value unknown.
2. `[?]` ADAU1777 digital-mic input availability — decides whether mic 5 needs an MCU PDM port.
3. `[?]` Whether the ADAU1777's programmable filters can be updated **glitch-free at 62.5 Hz** — the whole L2→L0 mechanism depends on it. Look for coefficient double-buffering / safeload.
4. `[?]` Alif E1C and CSK6012 unit pricing and Indian distribution.
5. `[?]` Measured NRR of the candidate cup — the single largest lever on system performance.
6. `[?]` IM73A135 India availability and minimum order quantity.
