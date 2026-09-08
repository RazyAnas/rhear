# Phase 0 — step by step, assuming no prior electronics experience

Written for someone who has never wired a microcontroller. Every step says what
to do, what you should see, and what to do when you do not see it.

**Do the milestones in order.** Each one proves something the next depends on.
Skipping ahead means debugging three things at once, which is how people lose
weekends.

---

## What you have (bills 2917, 2921, 2935 + earmuffs)

| item | qty | used in |
|---|---|---|
| ESP32-S3 N16R8 dev board | 1 | everything |
| INMP441 I2S mic module | 1 | Milestone 2-4 |
| MAX4466 mic amplifier module | **8** | Milestone 5 |
| 16 Ω 0.25 W 35 mm speaker | 2 | Milestone 6 |
| 830-tie breadboard | 1 | everything |
| 10 µF 63 V capacitor | 2 | power decoupling |
| 0.1 µF capacitor | 2 packs | decoupling |
| Capacitor box (assorted) | 1 | check it for 18-22 pF |
| Perfboard 9×15 cm | 1 | Milestone 5 mic jig |
| Berg strips M / F / 90° | 1 ea | soldering headers |
| Single-strand wire | 1 | jumpers |
| AT24C256 EEPROM | 1 | **Phase 1 only** — set aside |
| 3M Peltor X3A earmuffs | 1 | Milestone 6+ |
| ADAU1772 | on the way | **Phase 1** |

### Still missing

| item | needed for | note |
|---|---|---|
| **Resistors** (100 Ω, and an assortment) | speaker output | **nothing on any bill has resistors.** ~₹120 for a kit |
| 12.288 MHz crystal + 2× 18-22 pF | Phase 1 | check the capacitor box for the small values |
| MAX98357A I2S amplifier | nicer audio out | optional; PWM works without it |
| SPW2430 analog MEMS ×4 | Phase 1 final mics | MAX4466 is fine for all measurements |
| RT9013-33GB LDO | Phase 1 clean 3.3 V | |
| 40 mm 32 Ω drivers ×2 | the real headset | the 35 mm toy speakers are a measurement source only |

**You can complete Milestones 1-5 with what is already on your desk.**

---

# MILESTONE 1 — prove the board is alive  (~30 min)

## 1.1 Install the software

1. Download **Arduino IDE 2.x** from https://www.arduino.cc/en/software and
   install it.
2. Open it. Go to **File > Preferences**.
3. In *Additional boards manager URLs* paste:
   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```
   Click OK.
4. Go to **Tools > Board > Boards Manager**, search `esp32`, install
   **esp32 by Espressif Systems**. It is a large download; let it finish.

## 1.2 Select your board

**Tools >** and set:

| setting | value |
|---|---|
| Board | **ESP32S3 Dev Module** |
| USB CDC On Boot | **Enabled** ← important, or Serial prints nothing |
| Flash Size | **16MB (128Mb)** |
| PSRAM | **OPI PSRAM** ← your board is N16**R8** |
| Partition Scheme | 16M Flash (3MB APP/9.9MB FATFS) |
| Port | whatever appears when you plug the board in |

## 1.3 Blink

**File > Examples > 01.Basics > Blink**, then press **Upload** (the arrow).

**What you should see:** compile messages, then `Writing at 0x...`, then an LED
blinking on the board.

**If nothing happens:**
- No port listed → try a different USB cable. **Many cables are charge-only.**
  This is the single most common problem.
- Upload fails → hold the **BOOT** button, tap **RESET**, release BOOT, upload again.
- Still nothing → some S3 boards have two USB ports. Try the other one.

**Do not continue until the LED blinks.** Everything later assumes upload works.

---

# MILESTONE 2 — capture sound with the INMP441  (~2 hours)

## 2.1 Wire it

Power the board **off** (unplug USB) while wiring.

| INMP441 pin | ESP32-S3 pin |
|---|---|
| VDD | **3V3** (NOT 5V — 3.3 V only) |
| GND | GND |
| SCK | **GPIO 4** |
| WS | **GPIO 5** |
| SD | **GPIO 6** |
| L/R | **GND** |

Push both boards into the breadboard so their pins are in different rows, then
run single-strand wire between them. **L/R to GND** selects the left channel —
without it the mic may output silence.

**Avoid GPIO 0, 3, 45, 46.** Those are strapping pins and will stop the board
booting if held at the wrong level.

## 2.2 Test sketch

New sketch, paste this, upload:

```cpp
#include <driver/i2s.h>

#define I2S_SCK 4
#define I2S_WS  5
#define I2S_SD  6
#define SAMPLE_RATE 16000

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("starting i2s");

  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false
  };
  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num  = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = I2S_SD
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

void loop() {
  int32_t buf[256];
  size_t n = 0;
  i2s_read(I2S_NUM_0, buf, sizeof(buf), &n, portMAX_DELAY);

  long peak = 0;
  int samples = n / 4;
  for (int i = 0; i < samples; i++) {
    long v = buf[i] >> 14;              // 32-bit slot -> usable range
    if (labs(v) > peak) peak = labs(v);
  }
  Serial.println(peak);                  // loudness, one number per block
}
```

Open **Tools > Serial Monitor**, set baud to **115200**.

**What you should see:** a stream of numbers. **Quiet room = small numbers.
Clap or talk = big numbers.** That is a working microphone.

**If the numbers are always 0:**
- L/R not connected to GND
- SD / SCK / WS swapped — re-check against the table
- VDD on 5 V instead of 3.3 V

**If the numbers are always huge and never change:** SD pin is floating — the
wire is not making contact. Push it in properly.

**Use the Serial Plotter** (Tools > Serial Plotter) to see it as a live graph.
Talking should draw visible bumps. **This is your first real result — the
microphone works.**

---

# MILESTONE 3 — get real audio onto your laptop  (~1 hour)

Loudness numbers prove wiring. They do not prove the audio is *clean*. You need
to hear it.

## 3.1 Stream raw samples

Change `loop()` to send the raw samples as binary instead of printing peaks:

```cpp
void loop() {
  int32_t buf[256];
  size_t n = 0;
  i2s_read(I2S_NUM_0, buf, sizeof(buf), &n, portMAX_DELAY);

  int samples = n / 4;
  for (int i = 0; i < samples; i++) {
    int16_t s = (int16_t)(buf[i] >> 14);   // to 16-bit
    Serial.write((uint8_t*)&s, 2);         // little-endian
  }
}
```

Also change `Serial.begin(115200)` to **`Serial.begin(921600)`** — 16 kHz × 16
bits needs the higher speed.

## 3.2 Record it on the laptop

Close the Serial Monitor first (only one program can hold the port).

```bash
pip install pyserial numpy soundfile
```

```python
# record.py
import serial, numpy as np, soundfile as sf
PORT = "COM5"          # Windows: check Device Manager. Mac: /dev/cu.usbmodem*
SECONDS = 10
ser = serial.Serial(PORT, 921600, timeout=2)
ser.reset_input_buffer()
need = 16000 * 2 * SECONDS
data = bytearray()
print("recording... talk now")
while len(data) < need:
    data += ser.read(need - len(data))
x = np.frombuffer(bytes(data[:need]), dtype="<i2").astype(np.float32) / 32768.0
sf.write("capture.wav", x, 16000)
print("wrote capture.wav  peak", abs(x).max())
```

Run it, talk, then **play `capture.wav`**.

**What good sounds like:** your voice, clearly, maybe quiet. Some hiss is fine.

**What bad sounds like and why:**
- *Clicking / stuttering* → samples dropped. Lower the sample rate to 8000 in
  both the sketch and the script, confirm it is clean, then work back up.
- *Very quiet* → change `>> 14` to `>> 11` (louder). Do not go so far that loud
  sounds flatten out.
- *Buzzing* → mic wires too long or running beside the USB cable. Shorten and separate.

**Do not move on until `capture.wav` sounds like you.** Every later problem will
otherwise look like a model problem when it is really a wiring problem.

---

# MILESTONE 4 — run the AI on real audio  (~1 hour)  ← FIRST DEMO

The model does not need to run *on the chip* to prove the idea. Run it on your
laptop over audio the real hardware captured. That is a complete, honest,
demonstrable pipeline: **real mic -> real chip -> real model -> clean speech.**

```bash
cd ~/PS#2/E03
python3 - <<'PY'
import sys, os, torch, soundfile as sf, numpy as np
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "model")
from train_interim import enhance, N_FFT
from eval_stratified import load_model

x, fs = sf.read("capture.wav", dtype="float32")
assert fs == 16000, f"expected 16 kHz, got {fs}"
model, _ = load_model("runs/g7_hop256_50k/best.pt")
print("hop", int(model.hop_.item()), "| params",
      sum(p.numel() for p in model.parameters()))

win = torch.hann_window(N_FFT)
with torch.no_grad():
    _, est, _, _ = enhance(model, torch.from_numpy(x)[None], win)
sf.write("capture_enhanced.wav", est[0].numpy(), fs)
print("wrote capture_enhanced.wav")
PY
```

Play `capture.wav` then `capture_enhanced.wav` back to back.

**Record this comparison.** It is your first hardware-in-the-loop demo, and it
is worth more in a presentation than any simulation, because the audio came off
a microphone you wired yourself.

Try it with background noise — a fan, a video of an engine — to make the
difference audible.

---

# MILESTONE 5 — measure coherence  (~2 hours)  ← REAL SCIENCE

This one produces a number nobody on the team has: it tells you the **hard
physical ceiling** on how much noise the finished headset can ever cancel.

The rule is `NR <= -10 * log10(1 - coherence)`. At coherence 0.95, **no
algorithm anywhere can exceed 13 dB.** Our documents assume a value. You are
about to measure it.

## 5.0 What this actually has to measure — read this first

The coherence that bounds ANC is between the **reference mic outside the cup**
and **the noise at the ear, inside the cup**. The question the physics asks is:
*how well does the reference signal predict what reaches the eardrum?* Two mics
7 cm apart **in free air** answer a different and easier question, because there
is no earcup shell in between to decorrelate them.

**So the real measurement goes on the earcup.** Free air is a rehearsal, not the
result.

## 5.1 Rehearsal — the perfboard jig (1 hour, optional but recommended)

Solder **two MAX4466 modules onto the perfboard, exactly 7 cm apart**, centre to
centre, and measure in open air.

**Its only job is to prove your instrument works before you use it.** You will
be debugging ADC sampling, gain matching and the coherence script all at once,
and in free field you *know* the answer should be high. If it is not, the fault
is in the measurement, not the acoustics. Once free air gives a sensible number,
you can trust what the cup tells you.

Rigidity matters here: if the mics move between runs the numbers are
meaningless. That is what the perfboard buys you over the breadboard.

## 5.1b The real measurement — on the earcup  ← THIS IS THE RESULT

**Do not drill the muffs.** Not yet, and not for this.

- **Reference mic** — tape it to the **outside** of the cup shell
- **Error mic** — place it **inside** the cup, threading the thin wire **under
  the ear cushion**

The wire breaks the seal slightly and costs a little passive attenuation. That
is fine for a coherence measurement and it is completely reversible.

Aim for roughly **7 cm between them through the shell** — that spacing is where
the design's 204 µs of acoustic travel time comes from, and hence the 146 µs
causality budget.

**Why not drill:** every hole costs the NRR you paid for, and you cannot choose
hole positions sensibly until the 40 mm driver is in hand and you know where it
sits. Drilling is a Phase 2 decision.

**Report the CUP number, not the free-air number.** Free air will look better
and it is not your system.

## 5.2 Wire

| MAX4466 | ESP32-S3 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| OUT (mic 1) | **GPIO 1** |
| OUT (mic 2) | **GPIO 2** |

GPIO 1-10 are ADC1. **Do not use ADC2 — it stops working when WiFi is on.**

Set both modules to the **same gain**: point them at one steady sound and turn
each trimmer until the readings match within a few percent.

## 5.3 Sample both

```cpp
void setup() {
  Serial.begin(921600);
  analogReadResolution(12);
  analogSetPinAttenuation(1, ADC_11db);
  analogSetPinAttenuation(2, ADC_11db);
}
void loop() {
  uint16_t a = analogRead(1), b = analogRead(2);
  Serial.write((uint8_t*)&a, 2);
  Serial.write((uint8_t*)&b, 2);
  delayMicroseconds(125);          // ~8 kHz
}
```

Record ~20 s with broadband noise playing about a metre away (radio static, a
fan, a noise video).

## 5.4 Compute it

```python
import serial, numpy as np
from scipy.signal import coherence
PORT, FS, SECONDS = "COM5", 8000, 20
ser = serial.Serial(PORT, 921600, timeout=2); ser.reset_input_buffer()
need = FS * 4 * SECONDS
d = bytearray()
while len(d) < need: d += ser.read(need - len(d))
v = np.frombuffer(bytes(d[:need]), dtype="<u2").astype(float)
m1, m2 = v[0::2], v[1::2]
m1 -= m1.mean(); m2 -= m2.mean()
f, Cxy = coherence(m1, m2, fs=FS, nperseg=1024)
for lo, hi in [(100,500),(500,1000),(1000,2000),(2000,4000)]:
    s = (f>=lo)&(f<hi); c = Cxy[s].mean()
    bound = -10*np.log10(max(1-c, 1e-6))
    print(f"  {lo:5d}-{hi:5d} Hz   coherence {c:.3f}   max cancellation {bound:5.1f} dB")
```

**How to read it:** the last column is the most cancellation physically possible
at that frequency **with this microphone spacing**, no matter how good the
software is.

- **Above 15 dB at low frequencies** → geometry is fine, the design's claims hold
- **Below 10 dB** → the ceiling is lower than our documents assume, and that is
  a genuine finding you should report rather than hide

Expect the **cup** number to be lower than the free-air number. That is not a
mistake — the shell decorrelates the two signals, and that loss is exactly what
the reference microphone's placement has to be designed around. If the cup
number is much lower, moving the reference mic closer to the cup (shorter
acoustic path, less decorrelation) is the lever to try, at the cost of causality
margin. That trade is worth measuring rather than guessing.

Either way this belongs in the presentation. Very few teams measure their own
physical limit.

---

# MILESTONE 6 — sound output  (needs resistors)

**Buy 100 Ω resistors first.** Nothing on your bills has resistors, and you must
not connect the speaker straight to a pin: 3.3 V into 16 Ω is 206 mA and 0.68 W,
over both the pin's limit and the speaker's 0.25 W rating. **You will destroy
the GPIO, possibly the board.**

Wiring: `GPIO 7 -> 100 Ω -> speaker -> GND`, with a 0.1 µF capacitor across the
speaker (you have those).

```cpp
void setup() {
  ledcSetup(0, 100000, 8);     // 100 kHz carrier, 8-bit
  ledcAttachPin(7, 0);
}
void loop() {                   // 440 Hz test tone
  static float p = 0;
  p += 2*3.14159f*440.0f/16000.0f;
  if (p > 6.28318f) p -= 6.28318f;
  ledcWrite(0, (uint8_t)(127 + 100*sinf(p)));
  delayMicroseconds(62);
}
```

A quiet 440 Hz tone means it works. It will be faint — that is the 100 Ω doing
its job. **A MAX98357A module (~₹300) replaces this entire section** with a
proper amplifier and much better sound.

---

# What NOT to expect

**None of this does active noise cancellation.** The ESP32's ADC plus PWM output
is hundreds of microseconds; the ADAU1772 is 38 µs. Our own experiments measured
cancellation of impulsive noise falling to **exactly zero** at that latency.

That is not a failure of your build — it is the reason the ADAU1772 was chosen,
and being able to explain it is worth more than a half-working demo.

**Phase 0 proves:** the microphone chain, the AI model on real captured audio,
and the physical limits of the geometry.
**Phase 1, with the codec, proves:** the cancellation itself.

---

# BUILD IT SO THE CODEC DROPS IN  ← read before you wire anything

The point of this section: when the ADAU1772 arrives you should **plug it in**,
not rebuild. That requires two purchases now and one wiring discipline.

## ORDER TODAY — the codec cannot touch a breadboard without this

The ADAU1772 is a **40-pin LFCSP (QFN), 6×6 mm, 0.5 mm pitch, with a thermal pad
underneath.** There are no legs. It physically cannot go into a breadboard or
perfboard.

**You need a QFN-40 to DIP breakout adapter board.** Search terms:

```
QFN40 to DIP adapter 0.5mm pitch 6x6mm
QFN-40 breakout board 0.5mm
QFN/LQFP to DIP SMD adapter board
```

Order **3** (one per codec). They are ₹50-150 each on Robu / Evelta / Amazon.
Solder the codec to the adapter, solder header pins to the adapter, and it plugs
into the breadboard like any other chip.

**Without this the codec arrives and sits in its bag.** This is the single most
time-critical thing in this document.

## ARRANGE NOW — someone with a hot air station

A 40-pin QFN with a **thermal pad underneath the chip** cannot be soldered with
a normal iron. The pad has to be heated from below, which needs hot air or
reflow. This is not a beginner solder job and it is not a skill worth learning
on a ₹1,597 part with a 41-week lead time.

**Find the person or shop before the chip arrives.** Options in order:

1. A local electronics/repair shop with a **hot air rework station** — mobile
   repair shops usually have one, and this is a five-minute job for them
2. Your college lab, if it has reflow or hot air
3. A PCB assembly service

Take the adapter board, the chip, flux and solder paste. Ask them to reflow it.

## The pin plan — assign these on day one

**Reserve the codec's pins now and do not use them for anything else.** Then
Phase 1 adds wires instead of moving them.

| function | ESP32-S3 GPIO | phase | notes |
|---|---|---|---|
| INMP441 SCK | **4** | 0 and 1 | I2S peripheral **1** |
| INMP441 WS | **5** | 0 and 1 | stays forever |
| INMP441 SD | **6** | 0 and 1 | |
| MAX4466 mic 1 | **1** | 0 only | ADC1 |
| MAX4466 mic 2 | **2** | 0 only | ADC1 |
| PWM speaker | **7** | 0 only | |
| **codec MCLK** | **8** | 1 | **reserve** |
| **codec I2C SDA** | **9** | 1 | shared with the AT24C256 |
| **codec I2C SCL** | **10** | 1 | |
| **codec RESET** | **11** | 1 | **reserve** |
| **codec I2S BCLK** | **12** | 1 | I2S peripheral **0** |
| **codec I2S LRCLK** | **13** | 1 | |
| **codec I2S data OUT** (ESP -> codec) | **14** | 1 | |
| **codec I2S data IN** (codec -> ESP) | **21** | 1 | |

**The INMP441 stays on I2S1 and the codec gets I2S0.** The ESP32-S3 has two
independent I2S peripherals, so both run at once and neither has to move.

**Never use GPIO 0, 3, 45, 46** (strapping) or **33-37** (your N16R8's octal
PSRAM). Using those will stop the board booting, and the symptom looks like a
dead board.

## The wiring rule that saves the rework

> **Anything that will change in Phase 1 must end at a jumper or a header —
> never at a solder joint.**

Applied concretely:

- **The 7 cm mic jig**: solder the two MAX4466s to the perfboard (rigidity is
  the point), but bring their **OUT, VCC and GND to a 3-pin male header each**.
  Phase 0 those headers jumper to GPIO 1 and 2. Phase 1 the *same headers*
  jumper to the codec's analog inputs. **The jig never gets re-soldered.**
- **The speaker**: put it on a 2-pin header too. Phase 0 it hangs off PWM +
  100 Ω; Phase 1 it moves to the codec's DAC output.
- **Power**: build a proper rail. Feed the breadboard's + and - rails from the
  ESP32's 3V3 and GND *once*, and take every module's power from the rails.
  Adding the codec then means one wire to +, one to -, not re-doing power.

## Breadboard layout — leave the gap now

```
   [ + rail ]  <- 3V3 from ESP32, 10 uF + 0.1 uF across the rails
   ------------------------------------------------------------
   |  ESP32-S3 dev board (straddles the centre channel)        |
   ------------------------------------------------------------
   |  INMP441        (I2S1 - never moves)                      |
   ------------------------------------------------------------
   |                                                           |
   |   >>> LEAVE 20+ ROWS EMPTY HERE FOR THE CODEC ADAPTER <<< |
   |   the QFN adapter is wide; it needs the centre channel    |
   |                                                           |
   ------------------------------------------------------------
   |  AT24C256 EEPROM  (I2C - wire it in Phase 1)              |
   ------------------------------------------------------------
   [ - rail ]  <- GND
```

The empty block is the whole trick. **An 830-tie breadboard has room for all of
this if you plan it, and none of it if you do not.**

## Decoupling — do it from the start

You have 10 µF and 0.1 µF. Put **one 0.1 µF right beside each module's power
pins**, and the 10 µF across the rails near where power enters. Do it now, not
when problems appear — QFN codecs are far more sensitive to supply noise than an
INMP441, and adding decoupling later means disturbing working wiring.

## Phase 1, when the chip arrives — the whole checklist

1. Reflow the codec onto the QFN adapter (arranged above)
2. Solder headers to the adapter
3. Solder the **12.288 MHz crystal + two 18-22 pF caps** to the adapter's crystal pins
4. Plug the adapter into the reserved breadboard block
5. Two wires to the power rails
6. Seven wires to the reserved GPIOs (8-14, 21)
7. Move the mic-jig jumpers from GPIO 1/2 to the codec's analog inputs
8. Move the speaker jumper from PWM to the codec's DAC output
9. Wire the AT24C256 to the same I2C bus

**Nothing on this list requires unsoldering anything.** That is the goal.

## What you should NOT pre-wire

Do not guess the codec's pin assignments from memory — **read the datasheet's
pinout when the chip is in your hand** and check its supply rails (whether AVDD
and DVDD both want 3.3 V) before connecting power. Reserve the ESP32 side now;
decide the codec side from the document.

---

# If you get stuck

1. **Go back to the last milestone that worked.** Do not debug forwards.
2. **Check the USB cable first.** More projects stall on charge-only cables than
   on anything else.
3. **Unplug before rewiring.** Every time.
4. **Change one thing at a time**, and write down what you changed. It is the
   same rule that has caught nearly every bug in the software side of this
   project.
