# 09 — Phase-0 demo and the USP

What we can put in front of judges **today**, with no ADAU1772, and the
argument that makes it win rather than merely work.

Firmware: `hardware/rhear_demo/rhear_demo.ino` (+ `rhear_dsp.h`, `test_dsp.c`).
Hardware already on the bench: ESP32-S3-N16R8, INMP441, MAX98357A.

---

## 0. The one thing we will NOT claim

L0's causality budget is **146 µs**, computed from an assumed 7 cm
reference-mic-to-ear geometry against Kuo & Morgan's causality condition
(`docs/05-digital-twin.md:73-77`). Digital I2S MEMS microphones of the
INMP441's class carry several hundred microseconds of decimation-filter group
delay (`docs/04b-bom-verified.md:17-21`), which is why the design specifies a
low-latency codec instead.

**Stated precisely, because this will be probed:** we have not measured the
INMP441's group delay and no datasheet figure for it is quoted anywhere in this
repo. What is cited is the ADAU1772's own 38 µs at 192 kHz (datasheet Table 6).
`docs/04b-bom-verified.md:143-148` already records that the full ADC-to-DAC
latency of the chosen path "does not exist on paper and has to be measured on
hardware." So the correct claim is that the codec is specified for a budget the
digital-mic path is not designed to meet -- not that we measured the microphone
and found it wanting.

**We do not demonstrate live acoustic cancellation**, and we say so first,
unprompted.

What we demonstrate instead is the *actual L0 algorithm* — same tap count, same
FxNLMS update, same normalisation, running at 48 kHz on the target silicon with
its true per-sample cost printed on screen. Only the transducer path is
substituted. The codec is a component swap, not an architecture change.

Saying this out loud, before anyone asks, converts the missing part from a hole
into a credibility signal. Teams that overclaim get taken apart in Q&A.

---

## 1. The USP, in one sentence

> **The AI never touches the audio path — so you can kill the AI and hearing
> protection keeps working.**

Everything else follows from that.

### Why it is different

Most teams will put a neural network in the cancellation loop. That produces a
system where a model stall, a cache miss or a bad frame becomes **a hole in the
soldier's hearing protection**. Ours is architecturally incapable of that:

| layer | rate | what runs | in the audio path? |
|---|---|---|---|
| L0 | 48 kHz | FxNLMS, pure DSP, 20.8 µs/sample | **yes** |
| L1 | 62.5 fps | neural speech enhancer | radio path only |
| L2 | 62.5 Hz | scene engine — emits **coefficients**, not audio | **no** |

The AI sits in the *coefficient* path. It tunes the canceller; it never carries
the sound. That is the design decision the whole project is built on, and it is
demonstrable in four seconds with one keystroke.

### The second differentiator

> **Protection and communication are opposite problems, and we are the only
> ones treating them that way.**

- **Ear path** — remove machine noise, **keep human voices.** Suppressing a
  shouted warning is a safety failure, not a feature.
- **Radio path** — remove machine noise **and every voice that is not the
  wearer's.**

Same microphone, same instant, opposite targets. Audible A/B on the speaker.
This is a *requirements* insight, not a modelling trick, and it reads as domain
understanding rather than Kaggle skill.

### The third differentiator

> **We measured our own ceiling and published it against ourselves.**

A *perfect* 48-band magnitude mask scores **PESQ 2.392 at 0 dB** — below the
PS's own 2.5 target. So we can explain precisely why 100 GB of training data
moved PESQ by 1%, and why we stopped adding capacity instead of burning another
month on it. Most teams present only what worked.

---

## 2. Live demo script — five minutes

Flash `rhear_demo.ino`, open serial at 2000000 baud, put a noise source
(phone playing a generator, or a fan) near the mic.

| # | key | what happens | what to say |
|---|---|---|---|
| 1 | `P` | raw mic to speaker | "This is the microphone, untouched. This is the problem." |
| 2 | `N` `E` | ANC engages, `T` shows convergence climbing | "That is the real FxNLMS canceller adapting, live." |
| 3 | `T` | telemetry: µs/sample vs the 20.8 µs budget, headroom % | "Measured on this chip, not calculated in a slide." |
| 4 | `C` | radio path — same input, different output | "Same microphone. Opposite goal." |
| 5 | **`K`** | **"AI CORE KILLED" — cancellation continues** | **"I have just killed the AI. Protection is still running."** |
| 6 | `R` | AI returns, radio path recovers | "And it comes back with no glitch in the ear." |
| 7 | `A2.5` / `A1.0` | aggressiveness sweeps audibly | "One runtime knob, no retraining." |

**Step 5 is the demo.** Rehearse it. Pause after pressing `K`. Let the room
notice that the audio did not change.

### Then the recorded evidence (30 s)

Play from `E03/runs/nearfield/zero/`:

- `1_BEFORE.mp3` — you, plus a second person at 2 m, plus background
- `2_AFTER_mask_L1.mp3` — the other person gone
- `4_AFTER_plus_NEARFIELD_gate.mp3` — **silence between words**

---

## 3. The numbers to have on the slide

Every one of these is measured, and the file that produced it is in the repo.

| claim | number | where |
|---|---|---|
| **beats GTCRN's own published weights on our domain** | PESQ **1.716 vs 1.643**, STOI 0.824 vs 0.818, SI-SDR **10.76 vs 9.61** | `runs/g012/crossbench/gtcrn_dns3_edef.json` |
| **beats SepFormer at 1/516th the size** | **10.76 vs 5.56 dB** SI-SDR; 49,663 vs **25,613,569** params | `crossbench/sepformer_edef.json` |
| MetricGAN+ scores higher PESQ and *damages* the signal | PESQ 1.820, SI-SDR **−0.46 dB** | `crossbench/metricgan_edef.json` |
| L0 cost vs budget | printed live by `T` | on-chip telemetry |
| FxNLMS convergence | **+12.6 dB**, host-verified | `test_dsp.c` |
| kill the AI, protection survives | architectural | live |
| wearer vs bystander separation | **100%** (calibrated) | `nearfield_sim.py` |
| interferer suppression | **+10.1 dB** for 1.0 dB own-voice loss | docs/08 §6.4 |
| background between words | **−52.7 dB** | docs/08 §8.2 |
| L1 above 15 dB SNR | PESQ **2.607**, STOI **0.963** — both PS targets | docs/08 §2 |
| VoiceBank+DEMAND | STOI 0.928, SI-SDR 18.1 dB — 2 of 3 PS targets | docs/08 |
| our own proven ceiling | perfect 48-band mask = PESQ 2.392 at 0 dB | docs/08 §5 |
| simulation scale | 2,160 scenes, diffuse-field coherence modelled | `nearfield_sim.py` |

### Three "wow" lines that are all true

1. **"Our fifty-thousand-parameter model beats a twenty-five-million-parameter
   transformer on defence noise by 5.2 dB — and beats the published weights of
   the architecture we derive from, on all three metrics."**
2. **"You can kill our AI mid-demo and the ear protection doesn't flinch."**
3. **"We separate your voice from the person next to you using nothing but the
   curvature of the sound wave — no enrolment, no training on your voice, and
   it still works when you shout."**
4. **"We can tell you the exact PESQ score a *perfect* version of our model
   would get — 2.392 at 0 dB — which is how we knew to stop training and start
   changing the physics."**

---

## 4. Q&A preparation

**"Where is the codec?"**
Not arrived. L0's acoustic loop needs it because of a 146 µs causality budget
computed from the headset geometry, which the digital-MEMS-microphone path is
not designed to meet (we have not measured the INMP441 ourselves). Everything else — the algorithm, the timing, the
architecture, both signal paths — is running in front of you.

**"Is this just noise suppression?"**
No. Noise suppression is one of three layers. The other two are a real-time
canceller that never waits on a network, and a scene engine that retunes it.

**"Why not a bigger model?"**
We tried, and measured it: +56% parameters moved training PESQ by 0.001, and a
richer output head landed 1.4 PESQ below its own oracle. The limit is the
information in one microphone, not the network. That is why the next step is a
second microphone, not a bigger model.

**"Is it state of the art?"**
Within the constraint that matters — 200 KB, 125 MMAC/s, on an ESP32-S3 — the
architecture is a published ultra-low-complexity design and we are inside that
class. We are **not** at the top of the VoiceBank+DEMAND leaderboard and we do
not claim to be; we are specialised for impulsive defence noise, which is what
the PS asks for and what VoiceBank does not contain.

---

## 5. Build and verify

```
# prove the DSP before flashing anything
cd hardware/rhear_demo && clang -O2 -std=c11 -o /tmp/test_dsp test_dsp.c -lm && /tmp/test_dsp
```

That harness caught three real bugs that would each have wrecked the demo: an
NLMS step size that diverged at startup, a Hann-squared overlap-add that
violated COLA at 50% overlap, and a noise-floor tracker that ratcheted 30–100×
below the true noise so nothing was ever suppressed. None of them are visible
by reading the code; all three are obvious the moment the numbers are printed.
