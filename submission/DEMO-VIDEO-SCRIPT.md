# RHEAR — DEMO VIDEO SCRIPT
### 3 minutes. Every number in it is real and on your machine.

---

# ⏱ 0:00 – 0:15 — THE HOOK

**SCREEN: pure black. No logo. No title. Nothing.**

**AUDIO:** a gun line firing. Loud. Uncomfortably loud. Let it run 4 full
seconds — long enough that the viewer wants it to stop.

At 0:05, underneath the gunfire, a man's voice **shouting** — barely audible,
nearly buried:

> **"TAKE COVER —"**

**AUDIO CUTS DEAD. Total silence. Hold 1 second.**

**TEXT fades up, white on black, one line at a time:**

> He never heard it.
>
> **1 in 4** Indian Air Force personnel are losing their hearing.
>
> Permanently. It never comes back.

**VOICEOVER (calm, quiet — contrast against the noise):**

> "Ear defenders would have saved his hearing.
> They'd also have deleted the warning."

---

# ⏱ 0:15 – 0:45 — THE SOLUTION

**SCREEN: one simple animation. A soldier's head. ONE microphone. TWO arrows —
one to his EAR, one to his RADIO.**

**VOICEOVER:**

> "A soldier has two problems that pull in opposite directions.
>
> **His ears** must lose the engine, the rotor, the gunfire — and **keep** the
> human voice. Silencing a warning doesn't protect him. It kills him.
>
> **His radio** must lose the engine, the rotor, the gunfire — **and** every
> voice that isn't his. Otherwise his correction goes out with the gun line
> behind it.
>
> Same microphone. Same instant. **Opposite jobs.**
>
> Every noise-cancelling system in the world removes both.
> **RHEAR is built to know the difference.**"

**TEXT ON SCREEN, one line:**
> The AI tunes the canceller. It never carries the audio.

**VOICEOVER:**
> "Which means you can switch our AI off — and his ears stay protected.
> No software fault can ever deafen a soldier."

---

# ⏱ 0:45 – 2:15 — THE PROTOTYPE  (90 seconds, 5 shots)

## SHOT 1 — the hardware  (0:45 – 1:00, 15s)

**ACTION:** Hand-held close-up over the breadboard. Slowly point at each part.

**VOICEOVER:**
> "This is the whole thing. A ₹409 microcontroller. One microphone. One
> speaker. No GPU, no cloud, no internet."

📌 **What to show:** ESP32-S3 → INMP441 mic → MAX98357A + speaker. Point at each
as it's named.

## SHOT 2 — it's alive  (1:00 – 1:15, 15s)

**ACTION:** Screen recording of the serial monitor. Press **M**. Speak into the
mic. The bar moves.

**VOICEOVER:**
> "It's listening. And it's checking its own microphone every frame —
> 100% format valid, zero dropouts."

📌 Let the `[#########....] -22.4 dBFS` bar visibly jump when you talk.

## SHOT 3 — ⭐ THE MONEY SHOT — no voice, no sound  (1:15 – 1:40, 25s)

**ACTION:** Press **L** (live). Screen shows the `VOICE / --` line updating.

- **Talk for 5 seconds.** On screen: `VOICE ... gain 1.00`
- **Stop. Say nothing.** On screen: `-- ... gain 0.00`

**VOICEOVER, over the stop:**
> "Watch this. When I stop speaking —"

**AUDIO: cut the room tone to absolute digital silence for 2 seconds.**

> "— the channel doesn't go quiet. It goes to **zero**.
> Minus 120 decibels. That's a neural voice detector, running **on this chip**,
> deciding there is no human here."

📌 **This is the single best 10 seconds in your video.** Rehearse it. Pause
after "zero."

## SHOT 4 — hear the difference  (1:40 – 2:00, 20s)

**ACTION:** Press **A**. It records 5 s, then plays RAW, then plays PROCESSED.

**VOICEOVER (before it plays):**
> "Same five seconds. Twice. First raw, then through RHEAR."

📌 **Say NOTHING while the audio plays.** Put `RAW` / `RHEAR` as big text
labels on screen. Let the ears do the work.

## SHOT 5 — the proof  (2:00 – 2:15, 15s)

**ACTION:** Terminal. Type `./scorecard` and hit enter. Let the table appear.
Zoom into the comparison rows.

**VOICEOVER:**
> "We tested it against the best published models in the world, on 300 clips of
> real gunfire, shelling and rotor noise.
>
> **SepFormer** — a transformer with **25.6 million** parameters.
> **Ours** — forty-nine thousand.
>
> We beat it by **5.2 decibels**."

📌 Highlight the RHEAR row and the SepFormer row. Let `25,613,569` sit on screen
next to `49,663`.

---

# ⏱ 2:15 – 2:45 — THE IMPACT

**SCREEN: two numbers, huge, side by side.**

> **₹1,80,000**  ←  what the world's reference system costs
> **~₹20,000**  ←  what ours costs

**VOICEOVER:**

> "The system that solves this already exists. The US Army's TCAPS.
> It costs **₹1.8 lakh** a unit — and after years, it's reached twenty thousand
> soldiers. Its own price is the reason it stopped there.
>
> Ours **retrofits into ear defenders a unit already owns.**
> No new shell. No re-certification.
>
> At that price, you don't protect a pilot programme.
> **You protect the whole gun line.**"

**TEXT, final card:**
> No export-controlled parts.
> Trained in India. Weights never leave the country.
> **97 kilobytes that belong to us.**

---

# ⏱ 2:45 – 3:00 — TEAM

**SCREEN: faces, names, one line each. Fast cuts.**

> [Name] — model training and evaluation
> [Name] — embedded firmware and DSP
> [Name] — hardware and circuit design
> [Name] — data pipeline
> [Name] — systems and integration
> [Name] — testing

**FINAL FRAME, hold 3 seconds:**

> **RHEAR**
> It removes the man standing next to you from your radio.
> Not from your ears.

---

# 📋 SHOT PREP CHECKLIST

**Before you record:**

1. Flash `rhear_nsnet` and confirm the banner says **NSNET**, not NS_PRO
2. Run `L` once and tune the gate with `g` / `G` until it shuts cleanly when
   you stop talking — **do this before filming, not during**
3. Run `./scorecard` once so the terminal is warm and you know where the table
   lands on screen
4. Record the serial monitor with **screen capture**, not a phone pointed at a
   monitor
5. Set the serial monitor **font size large** — judges watch on phones

**Files you already have, if any live shot fails:**
- `E03/runs/nearfield/comms/1_RAW.mp3` and `3_G13b_NEURAL_GATED.mp3` — the A/B
- `E03/runs/nearfield/zero/` — the 8-second before/after
- `hardware/rhear_phase0_schematic.png` — the circuit, if you need B-roll

**Never debug on camera.** One take fails, cut to the recorded audio files.

---

# 🎯 THE THREE THINGS A VIEWER MUST REMEMBER

1. **A warning shout must reach his ears and must not reach his radio.**
   Nobody else makes that distinction.
2. **Kill our AI — his hearing protection keeps working.**
3. **₹1.8 lakh becomes ₹20,000, so everyone gets one, not a lucky few.**

If the video does nothing else, it has to land these.
