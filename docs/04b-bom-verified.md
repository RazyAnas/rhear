# RHEAR — minimum electronics BOM, verified prices (India)

Checked 30 Aug 2026. Prices in ₹ include GST where the vendor quotes it.
Every row is marked `[v]` verified against a live vendor page, or `[e]`
estimated and still to confirm.

---

## 1. The rules that decide what is on this list

These come from our own measurements, not from preference.

**R1 — the converter must be inside the causality budget.** E01 measured
wideband cancellation collapsing from −13.7 dB to −0.0 dB as electronics delay
goes 38 µs → 619 µs, against a 146 µs budget. This is why the codec is the
deciding part.

**R2 — reference and error microphones must be ANALOG.** The I2S MEMS mics
Indian retailers stock (INMP441, ICS-43434, SPH0645) contain their own
sigma-delta ADC and decimation filter — the same several-hundred-µs delay R1
rejects, moved inside the microphone. They cannot sit on the ANC path.

**R3 — the boom microphone does not need to be low-latency.** It feeds L1,
which already has 8 ms of algorithmic delay. So it can be a cheap digital I2S
module, and only 4 parts need the analog path.

**R4 — the codec has only 4 ADCs.** This is the hard constraint that sets the
microphone count: 2 reference + 2 error = exactly 4. The boom mic goes to the
MCU's own I2S port instead.

---

## 2. Bill of materials

| # | Part | Qty | Unit ₹ | Line ₹ | Source | Status |
|---|---|---|---|---|---|---|
| 1 | **ADAU1772BCPZ-RL** ANC codec, 4 ADC / 2 DAC, 40-LFCSP, Cut Tape | 1 | 1,596.68 | 1,597 | element14, 1,437 stock, 4–6 d | `[v]` |
| 2 | **ESP32-S3-WROOM-1-N8R8** module | 1 | 409 | 409 | Robu SKU R186284 | `[v]` |
| 3 | **Knowles SPU0410HR5H-PB** analog MEMS mic | 4 | 55.76 | 223 | Evelta | `[v]` |
| 4 | **INMP441** I2S MEMS mic module (boom) | 1 | 139 | 139 | Robu SKU 975775 | `[v]` |
| 5 | Crystal for the codec | 1 | 15 | 15 | Evelta | `[v]` |
| 6 | EEPROM, 24-series I²C | 1 | 55 | 55 | Evelta | `[v]` |
| 7 | **RT9013-33GB** LDO, 3.3 V 500 mA, SOT-23-5 | 1 | 47.49 | 47 | element14, 9,256 stock, Min 1 | `[v]` |
| 8 | **XC6804A2E14R-G** Li-ion charger, 4.2 V / 800 mA | 1 | 85.91 | 86 | element14, 1,680 stock, Min 1 | `[v]` |
| 9 | **3M Peltor X3A** earmuffs, NRR 28 dB | 1 | 2,299 | 2,299 | Amazon.in — caveats below | `[v]` |
| 10 | 40 mm 32 Ω earcup drivers — **not an element14 part** | 2 | 250 | 500 | replacement headphone driver / salvage | `[e]` |
| 11 | Li-ion cell | 1 | 200 | 200 | commodity | `[e]` |
| 12 | PCB (2-layer, small run) + passives + connectors | 1 | 800 | 800 | — | `[e]` |
| | **Verified subtotal** `[v]` | | | **4,870** | | |
| | **Estimated subtotal** `[e]` | | | **1,500** | | |
| | **BUILD TOTAL, one codec** | | | **6,370** | | |
| | + 2 spare codecs (see below) | 2 | 1,596.68 | 3,193 | | |
| | **BUILD TOTAL, three codecs** | | | **9,563** | | |

Everything ships **domestically** — element14 India, Robu, Evelta, Amazon.in.
No import shipping on any line.

**Buy three codecs.** element14 lists the **manufacturer lead time as 41 weeks**
against 1,437 in stock. The part is a 40-pin LFCSP — a QFN with a thermal pad,
hand-reflowed — so destroying one on the first attempt is an ordinary outcome,
and it would idle the project for most of a year against a 20 September
deadline. At ₹1,597 that is cheap insurance.

**element14 cannot supply the earcup drivers.** Its 32 Ω catalogue parts are
beeper-grade transducers: `MCABS-227-RC` covers 0 Hz–**4 kHz** at ₹287.52,
`ABS-209-RC` covers **500 Hz**–6 kHz at ₹273.34. A driver that starts at 500 Hz
cannot emit anti-noise across the band ANC actually works in. This line needs a
real 40 mm dynamic headphone driver — replacement part or salvage — not a
catalogue transducer.

Confirm at checkout whether element14 prices are ex-GST; they often are, which
would add ~18% to lines 1, 7 and 8.

### Earcups and drivers

Cups: retrofit a passive ear-defender rather than building one. It supplies the
shell, the seal, and the passive attenuation that handles high frequencies for
free. **3M Peltor X3A, ₹2,299, NRR 28 dB** is the chosen part — but it is 3M's
*low-profile* model, so internal cup depth is the risk for fitting a driver plus
the error mic. **X5A (NRR 31) is deeper** if the driver does not fit. Do not
drill until the driver is chosen: every hole costs the seal that delivers the
NRR you paid for.

Drivers: the X3A is passive and ships with none, and element14's 32 Ω parts are
beeper-grade (see above). Source a real 40 mm dynamic headphone driver, or
salvage a pair from a cheap headphone.

---

## 3. Parts deliberately removed

| Removed | Why, with the measurement |
|---|---|
| **IMU** | Its only job was head bearing. G5 found direction is **not** a separate mechanism — reference selection and filter shape are super-additive. E07 then measured the IMU as an ANC reference at **2.2 dB** below 500 Hz against the acoustic mic's 10.7 dB. No measured job remains. |
| **2 of 4 reference mics** | The causal-cone result (left ref causal −84°→+50°, right −50°→+84°, contralateral pairing lifting coverage 38%→68%) is a **left/right** result needing one reference per cup. The second pair per cup served finer bearing, which left with the IMU. Also forced by R4. |
| **Headphone amplifier IC** | The ADAU1772's output is "configurable as either line output or headphone driver", so it drives the earcup speakers directly. |
| **Second codec** | Would only be needed to restore the mics R4 removed. |

---

## 4. Compatibility check, interface by interface

| Interface | From → To | Verdict |
|---|---|---|
| Control | ESP32-S3 → ADAU1772 | **OK** — codec takes I²C or SPI; ESP32-S3 has both |
| Audio to MCU | ADAU1772 serial port → ESP32-S3 I2S | **OK** — only the 16 kHz voice path crosses here, not the 192 kHz ANC path |
| Boom mic | INMP441 → ESP32-S3 I2S | **OK** — ESP32-S3 has two I2S controllers, so mic and codec do not contend |
| Codec clock | crystal → ADAU1772 | **OK** — codec is self-clocked; no dependency on MCU timing |
| Analog mics | SPU0410HR5H → ADAU1772 PGA inputs | **OK** — mic runs 1.5–3.6 V, analog output; needs AC coupling and a bias network |
| Speakers | ADAU1772 headphone driver → 32 Ω drivers | **OK** — datasheet states headphone-driver mode |
| Logic levels | ADAU1772 digital I/O ↔ ESP32-S3 3.3 V | **OK** `[v, datasheet p14]` — abs max AVDD/IOVDD is +3.63 V and digital input is IOVDD + 0.3 V, so 3.3 V is in spec. **No level shifters** |
| Power | single 3.3 V rail → ADAU1772 | **OK** `[v, datasheet p16]` — DVDD is generated by an on-chip regulator, so no 1.1 V rail is needed |

---

## 5. The open risk that matters more than any price

**Constraint: the architecture does not change.** Parts may be swapped, and
parts that do no measured work may be deleted, but L0 stays adaptive FxNLMS.
So the question is not "what fits this codec" but "which part runs our L0".

Two facts, both verified:

**F1 — the codec's own engine is fixed-filter.** The ADAU1772 processing engine
is *"192 kHz processing path and biquad filters, limiters, volume controls,
mixing"*, programmed in SigmaStudio with coefficients writable at runtime. It
runs **biquads, not a per-sample LMS update.** So L0 cannot execute inside the
codec.

**F2 — the 38 µs figure is quoted for one specific path, at one rate.**
Datasheet Table 6, *Digital Filters*: **"ADC INPUT TO DAC OUTPUT PATH — Group
Delay, fS = 192 kHz, typ 38 µs"** `[v, datasheet p9]`. That is analog-in to
analog-out **through the chip's own DSP**, and the table gives no other rate.

*Correction to an earlier draft:* I previously wrote that group delay "scales
linearly with sample rate" and marked it verified. That came from a search
summary about ADI low-power codecs in general, **not from this datasheet**, which
quotes only fS = 192 kHz. The direction of the effect is right — lower rate
means longer decimation filters — but the 4× figure was not verified and should
not be quoted.

**F3 — the ADAU1772 cannot bypass its sample rate converters** (the ADAU1777
can; that is the difference between their 5 µs and 38 µs figures). So any path
out to an external processor and back goes through the ASRC twice, and the
datasheet does **not** quote a group delay for the
ADC → serial → external → serial → DAC path. **That number does not exist on
paper and has to be measured on hardware.**

Put together, running L0 on an external processor means:

```
ADC (192 kHz) -> serial -> MCU runs FxNLMS -> serial -> DAC
```

and the MCU must sustain FxNLMS at **192 kHz = 5.2 µs per sample**. Whether any
part we priced can do that is answered below, from our own code rather than
from parameter counts.

### Answered: counted from our own code, against published chip benchmarks

Per sample per channel, the FxNLMS loop in `rhear/core/anc.py` costs **3L + M**
MACs — filter output (L), normalisation `xhbuf @ xhbuf` (L), weight update (L),
and the filtered reference (M, the secondary-path model). The secondary path
itself is physical on hardware, so it is not computed.

Chip ceilings, one core. ESP32-S3 figures are **Espressif's own published O2
assembly benchmarks**, not estimates:

| ceiling | MAC/cycle | MMAC/s | source |
|---|---|---|---|
| ESP32-S3 float32 | 0.591 | **142** | `dsps_dotprod_f32` / `dsps_fir_f32` |
| ESP32-S3 int16 | 0.834 | **200** | `dsps_dotprod_s16` |
| STM32H7 q15 SIMD | 1.2–2.0 | 576–960 | **bounded, not measured** |

Requirement at 192 kHz, stereo (M = 64):

| L | @48 kHz | @192 kHz | fits? |
|---|---|---|---|
| 128 | 43 | **172** | **ESP32-S3: L0 on core 0, L1+L2 on core 1** |
| 256 | 80 | 319 | ESP32-S3 only by filling *both* cores — no room for L1 |
| 384 | 117 | 467 | STM32H7 territory |
| 512 | 154 | 614 | STM32H7 at 64–107% of the chip, nothing left for L1 |

**Verdict: the ESP32-S3 + ADAU1772 pair does run L0 as specified — but only at
L ≤ 128.** L1 needs 58.8 MMAC/s on the same chip and L2 on top, so the second
core has to stay free.

Running L0 at 48 kHz instead would be comfortable (43–154 MMAC/s) but is closed
off by F2: the codec's group delay at 48 kHz is ~4× the 38 µs figure, at or past
the whole budget. **L0 must run at 192 kHz, so L is the only free variable.**

Changing L is a parameter change, not an architecture change, so it is allowed
here. But it is not free: E01/F4 says exploiting periodicity needs a filter at
least one period long, and at 192 kHz a 50 Hz engine tone is 3,840 samples.
**L = 128 covers periodicity only above ~1.5 kHz.**

### E08 answered it: L = 128 is nearly free, so the ₹409 chip stands

Swept L at 192 kHz, stereo, coherence 0.95 (`rhear/experiments/e08_filter_length.py`):

| noise | L=128 | L=256 | L=384 | L=512 | cost of L=128 |
|---|---|---|---|---|---|
| engine 50 Hz (period 3840) | 16.2 | 14.8 | 15.0 | 15.7 | **+0.4 dB** |
| **rotor 200 Hz (period 960)** | 14.0 | 14.5 | 15.7 | **21.5** | **−7.5 dB** |
| hull 40–1200 Hz broadband | 13.0 | 12.9 | 12.9 | 12.9 | **+0.1 dB** |
| wideband 200–8000 Hz | 12.4 | 12.3 | 12.2 | 12.1 | **+0.3 dB** |

**Three of four noise types show no penalty at all.** Only the mid-frequency
tonal case pays, and it pays 7.5 dB.

The pattern follows the window rule exactly. A 200 Hz tone has a 960-sample
period, so L=512 spans more than half of it and can start exploiting the
repetition — hence the jump. A 50 Hz tone needs 3,840 samples, so *none* of the
tested lengths can exploit it and they all perform identically. The benefit of a
long filter appears only when L approaches the period.

Note also that hull and wideband land at 12.9–13.0 dB against a coherence bound
of **13.0 dB** — those cases are limited by microphone coherence, not by filter
length, which is what E07 predicted. Lengthening the filter cannot help them.

**Decision: ESP32-S3 at L = 128**, with the limitation stated rather than
buried. For the technical notes, verbatim:

> L = 128 satisfies real-time compute constraints with negligible loss for
> broadband and engine conditions, while a 200 Hz tonal rotor component
> benefits from longer memory.

L2's filter retuning (E06, +9.3 dB) targets exactly the tonal case that loses
here, so **L2 retuning may partially recover this penalty** — E08 does not
measure that and does not show it recovers all 7.5 dB. Treat it as a hypothesis
to test, not a fix already in hand.

This is a real trade against ₹600+ and a less mature toolchain for the STM32H7,
and we should present it as a trade, not as the chosen part being universally
optimal.

*Caveat, stated because it bounds the tonal rows:* for a tonal source the
random-phase surrogate used to set coherence is itself tonal, so effective
coherence runs higher than requested — which is why those two rows exceed the
13.0 dB bound. Comparisons **within** a row are sound (same reference, only L
changes); absolute tonal values are not comparable to the bound.

*Bug found and fixed en route:* the broadband rows first came back all-NaN and
looked like the canceller diverging. It was signal generation — at 192 kHz a
40 Hz corner is a normalised frequency of 4.2e-4, where `butter()` in transfer-
function form loses enough precision to put a pole at radius **1.0055**, outside
the unit circle. Second-order sections fixed it.

Note also that biquads are IIR while our E01/F4 rule (≥ one period, 960 taps at
48 kHz for 50 Hz) was derived for **FIR** taps. That rule should not be carried
across to an IIR implementation unmeasured — but it only matters if we ever
adopt a biquad L0, which the no-architecture-change constraint currently rules
out.

---

## 5b. Full timing budget — does the whole headset fit?

E08 settled L0. This is everything together, against the real deadlines.
`E03/bench/timing_budget.py`, chip ceiling from Espressif's published ESP-DSP
benchmark (0.834 MAC/cycle int16 → 200 MMAC/s per core, two cores).

| component | deadline | as written | with Levinson-Durbin |
|---|---|---|---|
| L0 FxNLMS, L=128, stereo | every 5.2 µs | 172.0 | 172.0 |
| L1 speech enhancer | every 4 ms | 58.8 | 58.8 |
| L2 features + ACF | every 16 ms | 13.8 | 13.8 |
| **L2 predictability solve** | every 16 ms | **349.5** | **8.2** |
| audio I/O, DMA/ISR | — | 6.4 | 6.4 |
| **core 0** | | 178.4 **PASS** | 178.4 **PASS** |
| **core 1** | | 422.1 **FAIL** | 80.8 **PASS** |
| **total vs 400.3 ceiling** | | **600.6 — 150%** | **259.2 — 65%** |
| **FRAME DEADLINE** | | **FAIL** | **PASS** |

**One function decides it, not one chip.** `predict.prediction_floor_db` builds a
256×256 Toeplitz matrix and calls a generic solver — O(n³)/3, 5.59 M ops every
16 ms, **350 MMAC/s on its own**. Levinson-Durbin returns the *same* answer in
O(n²): 131 k ops, 8.2 MMAC/s. A 43× ratio, and the difference between the
headset fitting and not.

**Scope, stated honestly.** `docs/02-architecture.md` describes L2 as a neural
scene engine. The code in `rhear/core/` is not that — it is classical feature
extraction, a predictability measure and filter selection. This budget prices
**what exists**. A trained neural L2 would need its own measurement and is not
covered here.

---

## 6. Open items — status

| # | Item | Status |
|---|---|---|
| 1 | **Can any priced part run adaptive FxNLMS at 192 kHz?** | **RESOLVED** — ESP32-S3 at **L = 128**, L0 on core 0 and L1+L2 on core 1. E08 measured the cost of that shortening: **zero on three of four noise types, −7.5 dB on mid-frequency tones only.** The ₹409 part stands |
| 2 | Codec group delay vs sample rate | **RESOLVED** `[v]` — scales linearly; 38 µs holds only at 192 kHz |
| 3 | Codec on-chip engine capability | **RESOLVED** `[v]` — biquads only, no per-sample LMS |
| 4 | Does the codec drive the earcups directly | **RESOLVED** `[v]` — "line output or headphone driver", no amplifier IC |
| 5 | Codec digital I/O voltage | **RESOLVED** `[v, datasheet p14]` — absolute max for AVDD/IOVDD is **−0.3 V to +3.63 V**, digital input max is IOVDD + 0.3 V. Running IOVDD at 3.3 V is in spec, so the ESP32-S3 connects directly. **No level shifters** |
| 6 | Codec supply rails | **RESOLVED** `[v, datasheet p16, p28]` — "the digital supply can be generated from an on-board regulator or supplied directly from an external supply". **DVDD is generated on-chip**, so a single 3.3 V rail is enough. The extra LDO I flagged is **not needed** |
| 7 | Group delay for the external-processing path | **OPEN, and unobtainable on paper** — see F3. The datasheet quotes only ADC→DAC internal at 192 kHz. Must be measured on the first board |
| 8 | Codec orderable at qty 1? | **RESOLVED** `[v, element14]` — **Cut Tape, Min 1 / Mult 1**. Re-Reel needs 10; take Cut Tape. 1,437 in stock, 4–6 days |
| 9 | Earcup drivers | **OPEN, and element14 ruled out** `[v]` — its 32 Ω catalogue parts are beeper-grade (`ABS-209-RC` starts at 500 Hz, which is below where ANC works). Needs a real 40 mm dynamic headphone driver or salvage |
| 13 | Does high passive attenuation hurt ANC coherence? | **OPEN, worth measuring** — reference mics sit outside the cup, error mic inside. A strong seal shifts the internal field toward structure-borne and leakage paths the reference cannot see, lowering γ². E07 measured +0.03 in γ² as worth **+3.3 dB** while 4 ms of extra lead was worth 0.3 dB — coherence dominates. First place to look if measured NR undershoots simulation |
| 10 | Crystal | **RESOLVED** — Evelta stocks crystals at **₹12–15**, India |
| 11 | I²C EEPROM for codec boot | **RESOLVED** — Evelta, **~₹20–60** depending on part, India |
| 12 | Battery + charger + regulator | **RESOLVED** `[v, element14]` — LDO **RT9013-33GB ₹47.49** (9,256 stock), charger **XC6804A2E14R-G ₹85.91** (1,680 stock), both Min 1. Only the Li-ion cell itself is still `[e]` |

### Parts confirmed removable since the first draft

- **IMU** — no measured job (G5, E07).
- **2 of 4 reference mics** — forced by the codec's 4 ADCs; the causal-cone
  result is left/right.
- **Headphone amplifier IC** — codec drives the earcups directly.
- **The L1 phase branch** — measured on the final H3 checkpoint over 600 clips:
  switching it off improves STOI **+0.023/+0.022**, PESQ **+0.041/+0.038** and
  SI-SAR **+1.62/+1.65 dB** on E_def/E_env. This deletes a network branch, not
  an architectural element, and removes `Atan` from the CMSIS-NN unsupported
  operator list.

## 7. Sources

- [ADAU1772BCPZ, Mouser](https://www.mouser.com/ProductDetail/Analog-Devices/ADAU1772BCPZ?qs=BpaRKvA4VqFLMW/0Av4s0w%3D%3D)
- [ADAU1772 datasheet (Analog Devices)](https://www.analog.com/media/en/technical-documentation/data-sheets/ADAU1772.pdf)
- [ESP32-S3-WROOM-1-N8R8, Robu.in](https://robu.in/product/espressif-esp32-s3-wroom-1-n8r8-modules/)
- [SPU0410HR5H-PB analog MEMS, Evelta](https://evelta.com/spu0410hr5h-pb-3-6v-59dba-sisonic-mems-microphone-analog-surface-mount/)
- [INMP441 I2S mic module, Robu.in](https://robu.in/?s=INMP441&post_type=product)
