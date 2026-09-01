# The RHEAR demo — what it is, how it works, how to present it

Everything explained in plain words, including every technical term.

---

## Start it

```bash
cd PS#2 && ./demo.sh
```

Then open **http://127.0.0.1:8765**. No internet needed. Stop it with Ctrl-C.

---

## Part 1 · What "digital twin" actually means

### The one-sentence version

> **A digital twin is a working software copy of a physical machine, running the
> same logic the real hardware will run, on the same signals, in real time — so
> you can watch it behave, and measure it, before the hardware exists.**

### Why we built one instead of buying parts first

The honest reason: **hardware is slow and expensive to be wrong with.**

A chip takes weeks to arrive and hours to solder. If we discover after all that
the design cannot work, we have lost a month. Our deadline is 20 September.

So we built the twin first, and it has already **rejected two components before
we spent money on them**:

- An ordinary audio chip — the twin measured that its delay reduces cancellation
  of gunfire and wind to **exactly zero**.
- The STM32F407 processor — the twin measured that our model needs 80–160% of it,
  and that 12 of its 24 required operations have no supported implementation.

Neither mistake cost us anything. That is the entire argument for the twin.

### What makes it a *twin* rather than an animation

This is the distinction a judge will probe, so be precise:

| | An animation / mock-up | Our digital twin |
|---|---|---|
| Where numbers come from | Written by hand to look good | Produced by the actual algorithm, running |
| Change the noise | Picture changes | The filter genuinely re-adapts |
| Can it be wrong? | No — it always looks right | **Yes, and it has been** |
| Can you measure it? | No | Yes — every number is logged |

**The last row is the important one.** A demo that cannot fail is not evidence.
Ours produces numbers we did not choose, and several of them have been bad — the
twin is where we discovered our direction hypothesis was wrong, our comb filter
did nothing, and our phase branch was actively harmful.

---

## Part 2 · How it mimics the hardware

The twin models **four separate physical things**. Each is a real measurement or
a real physical law, not a guess.

### 1. The microphones and their positions

Real geometry: reference microphones **outside** the cup, error microphones
**inside**, 7 cm apart. The twin computes how long sound takes to travel between
them — **7 cm of air is about 204 microseconds** — and delays the signals by
exactly that.

*Reference microphone* = hears the noise coming, so we can predict it.
*Error microphone* = hears what is left after cancelling, which tells the filter
how well it is doing.

### 2. The chip's delay

The real ANC chip (**ADAU1772**) takes **38 microseconds** to convert sound to
numbers and back. That number is from its datasheet. The twin inserts exactly
that delay, so what you see already includes the hardware's slowness.

An ordinary audio chip takes about **619 µs**. The twin can be run with either,
which is how we measured the difference.

### 3. The acoustic path from speaker to ear

Called the **secondary path**. Our anti-noise does not teleport to your eardrum —
it travels through a small speaker and a couple of centimetres of air, which
changes it. The twin models that, because the canceller must *compensate* for it.
That compensation is the "filtered-x" part of the algorithm's name.

### 4. Real recorded noise

Not synthetic hiss. Real recordings of gunshots, shelling, drones, helicopters,
fighter jets, vehicles, sirens and footsteps, mixed with real speech at
controlled loudness ratios, with **real measured room echo** from actual
recorded rooms.

### What the twin does NOT model — say this before you are asked

- **We have not built the board.** No soldering, no real microphone yet.
- **Room echo is applied from recordings**, not simulated from a 3D model of a
  tank interior.
- **The L1 speech enhancer is not in the live loop.** It runs offline; the A/B
  audio clips are its real output, but the live diagram does not run it.
- One number does not exist anywhere on paper and can only be measured on real
  hardware: the delay of the loop that goes out to the processor and back. The
  datasheet only gives the chip's internal figure.

Volunteering these is much stronger than being caught on them.

---

## Part 3 · Walking through the screen

### Top bar
Shows the **scene** (`rotor`, `wind`, `gunfire`…), whether it is live or paused,
and a time slider. The slider is worth using: **pause and scrub back** to a
moment where the noise changed, and show what the system did about it.

### Signal flow diagram (left)
The actual architecture. Watch the line colours:

- **Solid lines carry audio.**
- **Dashed lines carry coefficients — settings, not sound.**

**This is the single most important thing on the screen.** L2, the AI, only ever
touches dashed lines. It writes the filter settings; it never processes the
audio. That is why the AI never has to be fast enough to fit inside the
146-microsecond budget, and why the whole thing runs on a ₹409 chip.

### Microphones & signals (middle)
Live waveforms: what each microphone hears, the anti-noise going out, and the
**residual** — what is left at the ear after cancelling.

**The residual is the money shot.** It should be visibly smaller than the
disturbance above it. That is the system working, drawn from real numbers.

### State panel (right)
- **Attenuation** — how many dB of noise is being removed right now
- **Causality margin** — microseconds of headroom against the deadline. If this
  goes negative for an ear, that ear physically cannot be helped from that
  direction. Watch it change as the source moves
- **Filter norm** — how hard the filter is working
- **Compute** — block time and load, so the "does it fit" claim is visible live

### A/B panel (scroll down)
Eight real before/after audio pairs, one per defence noise class. **Play drone
or helicopter first** — those are the most audible.

### "What the audio cannot show you" (below the A/B)
The results that have no sound: the noise-removed vs damage-caused split, the
phase-branch finding, the filter-length decision, and the timing budget.

---

## Part 4 · How to present it in 60 seconds

1. **Point at the dashed lines.** *"Those carry numbers, not sound. The AI writes
   the filter settings sixty times a second and never touches the audio. That is
   the whole architecture."*
2. **Point at the residual waveform.** *"That is what is left at the ear. Real
   numbers from the real algorithm, not an animation."*
3. **Point at the causality margin.** *"Microseconds of headroom. When this goes
   negative, physics says that ear cannot be helped from that direction — and the
   system knows it."*
4. **Play the drone A/B clip.**
5. **Scroll to the findings.** *"And here is what you cannot hear: we remove
   20 dB of noise but net only 9.6, because distortion caps us. We do not have a
   noise problem — we have a speech-damage problem. Most teams never see this,
   because it only shows up if you split the score."*

Then stop talking. Let them ask.

---

## Part 5 · Every term used, explained

**Digital twin** — a software copy of a machine that runs the same logic on the
same signals, so you can measure it before building it.

**ANC** — Active Noise Cancellation. Play the opposite of the noise so the two
cancel at your ear. "Active" because it plays a sound; "passive" is just blocking
your ears with material.

**Microsecond (µs)** — one millionth of a second. Sound travels about 34 cm in
one millisecond, which is 1,000 microseconds.

**Latency / delay** — how long the electronics take. Our entire budget is
**146 µs**.

**Causality** — cause must come before effect. Our anti-noise must *arrive before*
the noise does. Miss it and you are adding noise, not removing it.

**Causality margin** — how many microseconds of that budget are left. Negative
means impossible, not merely difficult.

**Group delay** — how long a chip takes to pass sound through. The most important
hardware number in this project.

**Secondary path** — the route from our speaker to your eardrum. It changes the
anti-noise, so the canceller must compensate for it.

**FxNLMS** — Filtered-x Normalised Least Mean Squares. The maths in L0. It keeps a
list of numbers, uses them to build anti-noise, listens to what is left, and
nudges the numbers to make it smaller. *Filtered-x* = it accounts for the
secondary path. *Normalised* = it divides by how loud the input is, which turns
out to be a defence against gunshots, because a huge bang gets divided back down
instead of blowing the filter up.

**Coefficients** — the numbers in the filter. The AI writes these. This is the
entire mechanism by which the AI helps without touching sound.

**Attenuation** — how much quieter, in decibels. Every 10 dB is ten times less
power, so 20 dB is a hundred times.

**Residual** — what is left after cancelling.

**Coherence (γ²)** — how much two microphones hear the same thing, 0 to 1. It
caps what any algorithm can achieve: at 0.95 you can never cancel more than
13 dB. A limit on *microphone placement*, not on code.

**STOI** — 0 to 1, how *understandable* speech is. Target above 0.85. The one
that matters most.

**PESQ** — 1 to 4.5, how *natural* it sounds. Target above 2.5.

**SI-SDR** — in dB, how far speech stands above everything else. Target above 15.

**SI-SIR** — of that, how much noise was *removed*. Higher is better.

**SI-SAR** — of that, how much damage we *caused*. Higher means less damage.
Splitting SI-SDR into these two is how we discovered our real problem.

**MMAC/s** — millions of multiply-accumulate operations per second: the unit of
"how much maths per second". Our chip does about 200 million per core, and has
two cores.

**L0 / L1 / L2** — our three layers. L0 cancels noise 48,000 times a second with
no AI. L1 is a small neural network that cleans the outgoing voice 125 times a
second. L2 is the AI that watches the scene and re-tunes the other two 62 times
a second.

---

## Part 6 · Questions you will be asked

**"Is this simulated or real?"**
*"Simulated, and deliberately. Every number comes from the real algorithm running
on real recorded defence noise, with the real chip's 38-microsecond delay and the
real microphone geometry built in. What is not real is the board — we have not
soldered one. The simulator has already rejected two components before we spent
money, which is exactly why we built it first."*

**"How do I know the simulation is right?"**
*"Because it reproduces published results, including ones we initially got wrong.
And because it produces results we did not want — it is where we found out our
direction idea was not supported, our comb filter did nothing, and our phase
branch was hurting us. A simulator that only ever agrees with you is not being
run properly."*

**"Why not just build it?"**
*"We will. But a chip takes weeks to arrive and hours to solder, and the
manufacturer lead time is 41 weeks. Being wrong in hardware costs a month. Being
wrong in the twin costs an afternoon."*

**"Where is the AI in this picture?"**
*"The dashed lines. L2 writes filter coefficients sixty times a second and never
touches audio. That is the design decision the whole project rests on."*
