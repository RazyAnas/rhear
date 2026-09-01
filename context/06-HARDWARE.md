# Hardware

**Nothing has been built.** Everything below is verified-on-paper: parts priced
to order codes, budgets computed from our own code against published chip
benchmarks. Say this plainly — the simulator has already rejected two components
before money was spent, which is the argument for the twin.

## Architecture in one line

Reference mics **outside** the cup and an error mic **inside**; L0 cancels at
the ear; a separate boom mic feeds L1 whose output goes to the radio; L2 writes
coefficients to both and never touches audio.

**L1 has no path back into L0.** They are separate signal paths. (A diagram that
appeared to show otherwise was a layout bug — L2, L0 and L1 shared an
x-coordinate so the L2→L1 edge passed through the L0 box. Fixed in
`rhear/telemetry/schema.py`.)

## The physics that decides everything

- **Causality:** `tau_primary >= tau_secondary + tau_electrical`. With 7 cm mic
  spacing the budget is **146 us**. Miss it and you add noise instead of
  removing it. This is why the AI cannot sit in L0's audio path.
- **Coherence bound:** `NR <= -10*log10(1 - gamma^2)`. At gamma^2 = 0.95 you can
  never exceed 13 dB, whatever the algorithm. A microphone-placement limit, not
  a code limit.
- **Window rule:** exploiting periodicity needs a filter spanning >= one period.
  A 50 Hz engine fundamental is 960 samples at 48 kHz — E06 failed with L=256
  and worked immediately at L=1024. One experiment's rule predicted another's
  failure, which is the best evidence the twin models physics.

## Bill of materials — verified, ₹6,370

| part | qty | unit ₹ | note |
|---|---|---|---|
| **ADAU1772BCPZ-RL** ANC codec | 1 | 1,596.68 | element14, Cut Tape Min 1, **41-week lead time** |
| ESP32-S3 | 1 | 409 | dual core 240 MHz, 0.834 MAC/cycle int16 -> 200 MMAC/s/core |
| Knowles SPU0410HR5H-PB mic | 4 | 55.76 | analog MEMS |
| INMP441 | 1 | 139 | |
| crystal | 1 | ~15 | Evelta, India |
| I2C EEPROM (codec boot) | 1 | ~55 | Evelta, India |
| RT9013-33GB LDO | 1 | 47.49 | 9,256 in stock |
| XC6804A2E14R-G charger | 1 | 85.91 | 1,680 in stock |
| 3M Peltor X3A earmuffs | 1 | 2,299 | passive shell, NRR 28 dB |

**Total ₹6,370** with one codec, **₹9,563** with three (recommended: one to
solder, one for the mistake, one held back). All domestic.

## The number that chose the codec

**ADAU1772 group delay = 38 us.** An ordinary audio codec is **619 us**, and at
that latency gunfire cancellation measures **exactly zero**. The chip was chosen
from the physics, not the price list.

## The one number that exists nowhere on paper

The group delay of the **ADC → serial → processor → serial → DAC** loop. The
datasheet quotes only the internal ADC-to-DAC path at 192 kHz. This can only be
measured on real hardware and is the main reason to build.

## Rejected hardware

- **STM32F407** — needs 80–160% of the chip and **12 of 24 required operators
  have no supported implementation**.
- **NVIDIA Jetson AGX Orin** (which the PS suggests) — ~₹50,000 against our
  ₹409 processor. Our architecture makes it unnecessary because the AI writes
  coefficients rather than processing audio.

## Timing budget

See `02-RESULTS.md`. **263 of 400.3 MMAC/s, 66%, frame deadline PASS.**
At G7's operating point (hop 256) L1 drops 62.7 → 32.3 and core 1 goes
84.8 → 54.3.

## Deployment operating point vs demo

The live twin runs **48 kHz, L=1024** — algorithm validation. The deploy target
is **L=128 at 192 kHz**, forced by the codec's group delay. E08 measured the
cost of that shortening: **7.5 dB on a 200 Hz tonal rotor component, ~0 dB on
engine, hull and wideband noise.** Do not hide the 7.5 dB. Say "L2 retuning may
partially recover this penalty", never "L2 solves it".
