# Prototype Phase 0 — what to build before the ADAU1772 arrives

Parts in hand (Vellore Electronics, 01-09-2026, ₹2,009):
ESP32-S3 N16R8 · 1× INMP441 (I2S mic) · 4× MAX4466 (analog mic + amp) ·
2× 16 Ω 0.25 W 35 mm speaker · 9×15 cm perfboard · Berg strips · AT24C256
EEPROM · single-strand wire.

**Verdict: yes, this builds a meaningful prototype.** Not the ANC headset — that
needs the codec — but three things that de-risk the real build and one that is
worth more than the codec board itself.

---

## Read this first: what this hardware can and cannot do

| | possible now? | why |
|---|---|---|
| **Run L1 (the AI speech enhancer) on the real chip** | **YES** | L1 is the comms path at 16 ms frames. No 146 µs constraint. The ESP32-S3 in hand IS the deployment target |
| **Measure the real secondary path** | **YES** | speaker + mic is all it takes, and we currently only *model* this |
| **Measure inter-mic coherence at 7 cm** | **YES** | directly validates the bound that caps our whole design |
| **Real ANC cancellation** | **NO** | ESP32-S3 ADC + PWM output is hundreds of µs. Our own E01/E06 work showed cancellation of impulsive noise goes to **zero** at that latency. This is exactly why the ADAU1772 (38 µs) was chosen |

**Do not let Phase 0 quietly become "our ANC doesn't work."** It is not an ANC
rig. Frame it as: *Phase 0 proves L1 on target silicon and measures the acoustics
the twin has been assuming; Phase 1 (with the codec) proves L0.*

---

## Two hardware facts that will bite you

**1. The ESP32-S3 has NO DAC.** The original ESP32 had two 8-bit DACs; the S3
removed them. Audio out must be **PWM (LEDC)** through an RC filter, or I2S in
PDM mode. Plan for PWM.

**2. Do not drive the 16 Ω speaker straight from a GPIO.**
At 3.3 V into 16 Ω that is 206 mA and 0.68 W — over the pin's current limit and
over the speaker's 0.25 W rating. A blown GPIO takes the whole board with it.

### Parts still needed for audio OUT (not on the 01-09 bill)

**Option 1 — cheapest, ~₹200, uses what you have**

| part | value | qty | why |
|---|---|---|---|
| Resistor | **100 Ω** ¼ W | 4 | series with the speaker; protects GPIO and speaker |
| Ceramic capacitor | **100 nF** | 5 | across the speaker, smooths the PWM |
| Resistor assortment kit | — | 1 | ~₹120, needed constantly |
| Capacitor assortment kit | — | 1 | ~₹120 |
| Solderless breadboard, 830 pt | — | 1 | ~₹120. The 9×15 cm perfboard needs soldering for every change |

Circuit: `GPIO -> 100 Ω -> speaker -> GND`, 100 nF across the speaker. The
speaker's own inductance does most of the PWM filtering. Crude and quiet, but
adequate as a *sound source* — which is all Build B needs.

**Option 2 — ~₹300, recommended: MAX98357A I2S amplifier module**

An I2S **DAC + Class-D amplifier on one board**. It removes the whole problem:

- ESP32-S3 has no DAC -> it accepts I2S digital directly
- cannot drive 16 Ω from a GPIO -> it has a 3 W amplifier
- no RC filter, no series resistor, no PWM carrier to tune
- cleaner output means **better measurements** in Builds B and C

Wiring is BCLK / LRC / DIN plus power — the same pattern as the INMP441 already
in hand. **Buy this if the shop has it.**

*It does not enable ANC.* I2S DAC + Class-D is still hundreds of microseconds
against the ADAU1772's 38 µs. It improves audio quality, not latency.

---

## Build A — L1 on target silicon  ← DO THIS ONE FIRST

**The single most valuable thing available to you right now.** It converts
"our model fits the chip" from a spreadsheet claim into a measurement, and it is
the AI contribution the PS actually asks about.

**Wiring — INMP441 to ESP32-S3 (I2S):**

| INMP441 | ESP32-S3 | note |
|---|---|---|
| VDD | 3V3 | **3.3 V only** |
| GND | GND | |
| SD | GPIO 6 | serial data |
| WS | GPIO 5 | word select / LRCLK |
| SCK | GPIO 4 | bit clock |
| L/R | GND | selects left channel |

Any free GPIOs work; keep them away from strapping pins (0, 3, 45, 46).

**Steps**

1. Bring up I2S capture at **16 kHz, 32-bit slots** (INMP441 outputs 24-bit
   left-justified in a 32-bit slot — take the top 24 bits, then scale to float).
2. Record 10 s to flash or stream over USB serial. **Listen to it.** Confirm it
   is clean before touching the model — every downstream problem otherwise looks
   like a model problem.
3. Export the current best model to ONNX / TFLite-Micro. Current best is
   **G7**: `E03/runs/g7_hop256_50k/best.pt`, 49,663 params, hop 256,
   **32.3 MMAC/s at 62.5 frames/s**.
4. Run it frame by frame on captured audio. **Measure wall-clock time per
   frame.** Budget is **16 ms per frame** at hop 256.
5. Report actual vs modelled. Our figure of 32.3 MMAC/s assumes 0.834 MAC/cycle
   int16 from Espressif's published ESP-DSP benchmarks. **This is the first
   chance to check that assumption against reality**, and it is a number a judge
   can ask about.

**Success = a measured milliseconds-per-frame figure.** Even if it is slower
than modelled, that is a real result and it tells you exactly how much capacity
you can afford. If it comes in comfortably under 16 ms, you have hard evidence
the design fits, on the actual part, which nothing in the twin can give you.

---

## Build B — measure the secondary path

The twin models the speaker→ear acoustic path. **Nobody has measured it.** This
replaces an assumption with data, and it is the input FxNLMS needs.

**Wiring:** speaker on a PWM pin through **100 Ω** (and an RC low-pass, e.g.
1 kΩ + 100 nF, to smooth the PWM). One MAX4466 into an ADC pin (GPIO 1–10 are
ADC1; **avoid ADC2, it conflicts with WiFi**).

**Steps**

1. Mount the speaker and the mic a few cm apart, ideally inside an earcup once
   you have one.
2. Play a **logarithmic sine sweep**, 100 Hz → 8 kHz, ~2 s. Record the mic.
3. Deconvolve to get the impulse response. Compare with what
   `rhear/sim/` currently assumes.
4. Note the delay in samples — that is `tau_secondary`, a term in the causality
   inequality `tau_primary >= tau_secondary + tau_electrical`.

**Why it matters:** a measured secondary path can be dropped straight into the
twin, which makes every subsequent L0 result more credible than a modelled one.

---

## Build C — measure inter-mic coherence at 7 cm

**The highest-leverage measurement of the three**, because coherence is a hard
physical cap on everything L0 can ever achieve:

`NR <= -10 * log10(1 - gamma^2)`

At γ² = 0.95 no algorithm exceeds **13 dB**. We have been *assuming* a coherence
value. Measuring it tells you whether your microphone geometry — not your code —
is the limit.

**Steps**

1. Two MAX4466 mics **exactly 7 cm apart** (the reference-mic spacing in the
   design), both into ADC1 pins.
2. Sample both at ≥ 8 kHz. Play broadband noise from ~1 m away.
3. Compute magnitude-squared coherence vs frequency (Welch).
4. Convert to the NR bound at each frequency.

**If measured coherence is lower than assumed, the design's cancellation ceiling
is lower than our documents claim — and it is far better to learn that now than
after the codec board is built.** This is a result worth putting in the deck
either way.

---

## The MAX4466 caveat, and why 4 of them is right

MAX4466 is **manual gain** — good. (The MAX9814's automatic gain control would
have fought the adaptive filter, which assumes a stable acoustic path.)

**Set all four to the same gain and mark them.** Mismatched sensitivity between
reference and error mics costs coherence, and coherence is the cap above. Match
them by pointing all four at the same source and trimming until their recorded
levels agree within ~1 dB.

Also: these are **electret** modules, not the MEMS parts of the final BOM. Use
them for measurement and bring-up, not for final performance numbers.

---

## What each part in the bill is for

| part | Phase 0 use |
|---|---|
| ESP32-S3 N16R8 | the target processor. 16 MB flash / 8 MB PSRAM — plenty for audio buffers |
| INMP441 | **Build A** — clean digital capture, no ADC noise |
| MAX4466 ×4 | **Builds B and C** — analog mics for path and coherence measurement |
| 16 Ω speaker ×2 | **Build B** — sound source. **100 Ω in series**, or via a MAX98357A |
| perfboard + Berg strips | mount it so the mic spacing is rigid and repeatable — coherence measurements are worthless if the mics move |
| AT24C256 EEPROM | **not needed yet.** It is for the ADAU1772's boot configuration in Phase 1. Set it aside |
| single-strand wire | keep mic leads short; long unshielded analog runs pick up hum |

---

## Order of work

1. **Build A** — L1 on chip. Highest value, no acoustics needed, works on a desk.
2. **Build C** — coherence. One afternoon, and it can change the design's claims.
3. **Build B** — secondary path. Best done once you have an earcup.

When the ADAU1772 and the 12.288 MHz crystal arrive, Phase 1 begins and the
codec replaces the ADC/PWM path entirely. **Nothing built here is wasted:** the
measurements feed the twin, and the L1 timing figure stands on its own.

---

## What to say about it

> "We are running our speech-enhancement model on the actual ESP32-S3 today, and
> measuring the acoustic paths our simulator has been assuming. The active
> cancellation stage needs the ADAU1772's 38 µs converter — an ordinary ADC/PWM
> loop is hundreds of microseconds, which our own experiments showed drives
> impulsive-noise cancellation to zero. That part arrives this week."

Honest, specific, and it turns a missing part into evidence that you understand
why that part is required.
