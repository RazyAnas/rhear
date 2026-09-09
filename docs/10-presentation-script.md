# 10 — Presentation script (6 speakers, 10 minutes)

---

## THE ONE LINE

> ### "RHEAR removes the man standing next to you from your radio — without removing him from your ears."

Say this in the first 20 seconds. Say it again in the last 20.

Why it works: it is concrete, it is defence-specific, a non-expert gets it
instantly, and **no other team will have thought about it**, because it only
occurs to you once you realise protection and communication are *opposite*
problems. Everyone else is building "a denoiser."

### The three proofs behind it

| # | claim | proof |
|---|---|---|
| 1 | **Physics, not machine learning.** We tell your voice from his using the *curvature of the sound wave*. Your mouth is 5 cm from the boom; he is 2 m away. No voice enrolment, no training on you, still works when you shout. | 100% separation, +10.1 dB, `nearfield_sim.py`, 2,160 scenes |
| 2 | **The AI never touches the audio.** It tunes the canceller; it never carries the sound. Kill the AI mid-demo and hearing protection does not flinch. | live, one keystroke |
| 3 | **We measured our own ceiling.** We can tell you the score a *perfect* version of our model would get. That is how we knew to stop training and start changing the physics. | PESQ 2.392 at 0 dB, docs/08 §5 |

**Proof 3 is the one that wins technical judges.** Any team can say "our model
scores X". Almost none can say "and here is the mathematical maximum any model
of this class could score, here is how we computed it, and here is what we did
when we found we were near it."

---

## ROLES AND SCRIPT

Times are speaking times. Rehearse to them.

---

### 1 — THE PROBLEM (60 s)

> "A soldier in a vehicle bay has two problems that pull in opposite directions.
>
> He needs to **not go deaf** — that means killing the engine, the rotor, the
> gunfire.
>
> And he needs to **be understood on the radio** — that means his voice getting
> through, and nothing else.
>
> Here is what nobody builds for: **those two goals conflict.** If a man beside
> him shouts a warning, that voice must reach his ears — suppressing it is a
> safety failure. But that same voice must *not* go out over his radio, because
> the radio is his channel alone.
>
> One microphone. One instant. Two opposite requirements.
>
> **RHEAR removes the man standing next to you from your radio, without removing
> him from your ears.** That is what we built."

---

### 2 — ARCHITECTURE AND WHY IT IS SAFE (90 s)

> "Three layers, running at three different rates on one ESP32-S3.
>
> **L0** is hearing protection. 48 kHz, pure DSP, a 20.8 microsecond budget per
> sample. **No neural network anywhere near it.**
>
> **L1** is the speech enhancer for the radio. Neural, 16 millisecond frames.
>
> **L2** is the scene engine. It runs 62 times a second and it emits
> **coefficients** — numbers that retune L0. It never carries audio.
>
> That is the decision the whole project rests on. **The AI sits in the
> coefficient path, not the audio path.**
>
> Why that matters: every system that puts a network inside the cancellation
> loop has a failure mode where a stalled model becomes *a hole in the
> soldier's hearing protection*. Ours is architecturally incapable of that.
>
> [DEMO DRIVER PRESSES K]
>
> I have just killed the AI core. Listen. **Protection is still running.**"

---

### 3 — LIVE DEMO (2 min)

Rehearse. Know the failure modes. Have the recorded audio ready as a fallback.

| beat | key | say |
|---|---|---|
| 1 | `D` | "First, the system checks itself — amplifier, microphone format, loopback. It tells us what is wrong rather than making us guess." |
| 2 | `P` | "This is the raw microphone. This is the problem." |
| 3 | `N` `E` | "The canceller is adapting, live. Watch the dB climb." |
| 4 | `T` | "Microseconds per sample, measured on this chip against a 20.8 µs budget. Not a slide — the chip." |
| 5 | `C` | "Same microphone. Opposite goal. This is what the radio hears." |
| 6 | **`K`** | **[PAUSE 3 SECONDS]** "The AI is dead. The protection is not." |
| 7 | `W` | "And this is my own voice, recorded and cleaned on the device, played back to you." |

**If hardware fails:** "We have this recorded — here is the same thing" and play
`E03/runs/nearfield/zero/`. Never debug on stage.

---

### 4 — DATA AND TRAINING (90 s)

> "The PS asks for a scalable dataset pipeline. Ours generated over **100 GB**
> of noisy-clean pairs: **251 speakers**, **236 noise classes** including
> gunshot, shelling, rotor and jet, real measured room impulse responses, and
> explicit control of the signal-to-interference ratio.
>
> But the number I want to give you is a **negative** one.
>
> We went from 30 GB to over 100 GB of training data. PESQ moved about **one
> percent**. Most teams would hide that. We went and found out why — and the
> answer changed the project."

*(hand to speaker 5)*

---

### 5 — EVIDENCE, AND THE CEILING (2 min) — **the technical high point**

> "We asked a question most projects never ask: **what is the best score our
> model could *possibly* get, if it were perfect?**
>
> So we built the oracle. We gave the system the correct answer — the ideal
> mask, computed from the clean speech — and scored it.
>
> **A perfect 48-band mask scores PESQ 2.392 at 0 dB. The target is 2.5.**
>
> Our model was not underperforming. It was **climbing toward a ceiling that
> sits below the finish line.** That is why 100 GB moved us one percent.
>
> We confirmed it three ways. More parameters: **+56% gave +0.001 PESQ.** A
> richer output head: it landed **1.4 PESQ below its own oracle**. More data:
> one percent. Three independent levers, same answer.
>
> So we stopped adding capacity and changed the **information**.
>
> Your mouth is 5 centimetres from the boom microphone. Anyone else is metres
> away. In the near field, sound spreads spherically, and a second microphone
> a few centimetres further along the boom sees your voice **6 dB louder** and
> his voice **0.2 dB louder**. That is a thirty-fold difference in the cue.
>
> **100% separation. +10.1 dB of suppression for 1 dB of your own voice. No
> enrolment. Works when you shout.**
>
> And every one of those decisions was made by a paired bootstrap over ten
> thousand resamples, on two independent evaluation sets. **We rejected four of
> our own models that way**, including one we had already trained."

---

### 6 — HARDWARE, HONESTY, CLOSE (90 s)

> "The codec has not arrived. I will tell you exactly what that costs us,
> because you will ask.
>
> L0's acoustic loop needs a **146 microsecond** causality budget. The
> microphone's own decimation filter spends more than that before a single
> instruction runs. So **we are not going to stand here and claim acoustic
> cancellation we cannot do today.**
>
> What you saw was the real algorithm — same taps, same update, same timing,
> measured on this chip. The codec is a component swap, not an architecture
> change. The schematic is done and electrically verified: 37 parts, 30 nets,
> ERC clean.
>
> Where we stand against the problem statement: above 15 dB input SNR we meet
> **all three targets** — PESQ 2.607, STOI 0.963, SNR 18 dB. Below that, STOI
> holds and PESQ does not, and we have shown you exactly why.
>
> One microphone away from closing it.
>
> **RHEAR removes the man standing next to you from your radio — without
> removing him from your ears.** Thank you."

---

## WHAT TO DO WHEN ASKED

**"Is the neural model running on the ESP32?"**
> "L0 is, fully. L1's neural model runs on the host today; its on-chip port is
> partial and that is our next milestone. The DSP enhancer you heard on the
> device is real and measured, and we label it as a DSP enhancer, not as the
> network."

**"Is this state of the art?"**
> "Within the constraint that matters — 200 KB, 125 MMAC/s, one ESP32-S3 — we
> use a published ultra-low-complexity architecture and we are inside that
> class. We are **not** top of the VoiceBank leaderboard and we do not claim to
> be. VoiceBank is cafes and offices. We are built for gunfire and rotors."

**"Why not just use a bigger model?"**
> "We tried, and measured it. +56% parameters gave +0.001 PESQ. The limit is
> the information in one microphone, not the size of the network."

**"What is genuinely new here?"**
> "Three things. Using near-field wavefront curvature to separate the wearer
> from a bystander with no enrolment. Putting the AI in the coefficient path so
> hearing protection cannot be taken down by a model stall. And computing the
> oracle ceiling of our own model class, which is how we knew to stop training."

---

## RULES FOR THE ROOM

1. **Lead with the one line. Close with the one line.**
2. **Volunteer the missing codec before anyone asks.** Overclaiming is how
   teams get taken apart in Q&A; pre-empting it buys you credibility for
   everything else.
3. **Never debug on stage.** One failed attempt, then the recording.
4. **Pause after pressing `K`.** Three full seconds. Let them notice.
5. **Give numbers with their source.** "2.392, from the oracle analysis" beats
   "very high" every time.
