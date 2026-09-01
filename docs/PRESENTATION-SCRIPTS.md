# RHEAR — presentation scripts, all six members

SIH 2026 · PS 26052 · DRDO / iDEX · **10 minutes maximum, aim for 9.**

---

## Running order and time budget

| # | Slide | Speaker | Backup | Time |
|---|---|---|---|---|
| 1 | Title | **Manaswi** opens | — | 0:15 |
| 2 | Idea / Solution · Problem resolution · Unique value | **Manaswi** → **Faraz** | each other | 1:50 |
| 3 | Technical approach — architecture and layers | **Anas** | **Vishnu** | 2:45 |
| 4 | Feasibility and viability | **Suryansh** | **Faraz** | 2:00 |
| 5 | Impact and benefits | **Raj** | **Vishnu** | 1:40 |
| 6 | Research and references | **Raj** | **Anas** | 0:40 |
| | | | **total** | **≈ 9:10** |

**Backup means:** if a judge asks something your slide-mate cannot answer, you
take it. Do not interrupt otherwise.

**The one rule for everyone:** never say a number without saying what it is
measured on. "STOI 0.814 on 300 clips of real defence noise the model has never
heard" — not "STOI 0.814".

**The second rule, new and important:** our test set is deliberately brutal —
**a third of its clips sit below 0 dB SNR**, where the noise is louder than the
speech. Averages over it understate what the headset does in the field. Whenever
you give an average, be ready to give the breakdown, because the breakdown is
much stronger than the average.

---

# 1 · MANASWI — Idea / Solution (opens the deck)

## Your job in one line
Make the judges *feel* the problem in 30 seconds, then tell them what we built.

## What / Why / How

**WHAT we are building.** A military communication headset that keeps speech
understandable inside armoured vehicles, helicopters and gunfire.

**WHY it does not already exist.** Ordinary noise cancelling — the kind in
consumer headphones — assumes noise stays the same. Defence noise does not. It
changes every few seconds, and it contains 160 decibel bangs. A gunshot is not
"loud noise"; it is a different physical problem.

**HOW we solve it.** Three systems running at three different speeds. A fast,
simple one cancels noise 48,000 times a second. A small AI cleans the voice going
out over the radio. A second AI watches the environment and re-tunes the other
two sixty times a second. **The AI never sits in the fast loop — it writes the
settings, it does not touch the sound.**

## Pointers you must cover

1. **Open with the scene, not the tech.** "A soldier in a moving vehicle says
   *hold fire* over the radio. Engine noise, rotor thump and a gunshot hit the
   same microphone. The person listening hears noise — or worse, hears a word
   that was never said."
2. Say why classic methods fail: **they assume the noise sits still.**
3. Name the three targets we are judged on: **speech understandable (STOI) above
   0.85, sounds natural (PESQ) above 2.5, speech above noise (SNR) above 15 dB.**
4. **The unique value in one sentence:** *"The AI decides what the filter should
   be; it never has to be fast enough to sit in the audio path — which is why
   this runs on a ₹409 chip instead of a ₹50,000 one."*
5. Hand to Faraz with a line, not a pause: *"Faraz will show why that decision
   was forced on us by physics."*

## Study material — everything explained

**Noise cancellation, plainly.** Sound is pressure waves. If you play a wave that
is the exact opposite shape at the right moment, the two cancel and you hear
silence. That is Active Noise Cancellation, ANC. "Active" because it plays a
sound; "passive" is just blocking your ears with material.

**Stationary vs non-stationary noise.** Stationary means the noise keeps the same
character — a fan, a steady engine hum. Non-stationary means it changes — a
vehicle accelerating, a helicopter banking, gunfire. Almost all classic methods
were designed for stationary noise, which is why they struggle here.

**Impulsive noise.** A single very short, very loud event — a gunshot, a shell.
The problem is not just volume. Classic adaptive filters *learn* from what they
hear, and one enormous sample can make them learn something catastrophically
wrong. The filter can go unstable and start *adding* noise.

**The three scores, in plain words.**
- **STOI** — a number from 0 to 1 for *how understandable* speech is. This is the
  one that matters most in a defence radio: it predicts whether the listener
  hears the right word.
- **PESQ** — a number from 1 to 4.5 for *how natural* it sounds. Modelled on real
  human listening tests.
- **SNR / SI-SDR** — in decibels, *how far the speech stands above everything
  else.* Decibels are a ratio on a logarithmic scale: every 10 dB is ten times
  more power, so 20 dB is a hundred times.

**Why our value proposition is real.** Every competing approach we found puts the
neural network inside the audio path. That forces the network to finish its work
in under 146 microseconds — a millionth-of-a-second budget explained on the next
slide. On a cheap chip, that is not a small network; it is *no* network. By
letting the AI write settings instead of processing sound, we removed that
constraint entirely.

---

# 2 · FARAZ — the physics that forces the design (with Manaswi, slide 2)

## Your job in one line
Show that our architecture was not a preference — it was the only thing physics
allowed.

## What / Why / How

**WHAT the constraint is.** Our anti-noise has to reach the ear *before* the real
noise does. Sound travels from the outside microphone to the ear in a fixed time.
Everything electronic — microphone, chip, speaker — has to finish inside that
window. **The window is 146 microseconds.**

**WHY it decides everything.** We measured what happens when the electronics are
too slow. On unpredictable noise — gunfire, wind — cancellation goes from
**−13.7 dB to −0.0 dB** as the delay grows from 38 to 619 microseconds. Zero.
Nothing. An ordinary audio chip takes about 619 microseconds, so an ordinary chip
cancels nothing on exactly the noise defence cares about most.

**HOW we handled it.** We chose the chip *from the budget*, not from a feature
list. The ADAU1772 does it in 38 microseconds.

## Pointers you must cover

1. The causality rule in words: **anti-noise must arrive first, or you are adding
   noise instead of removing it.**
2. The measured cliff: **−13.7 dB at 38 µs → −0.0 dB at 619 µs.**
3. Why engine hum survives long delays but gunfire does not: **hum repeats, so
   you can predict it. Gunfire does not repeat.**
4. **One line that lands:** *"A normal audio chip is four times too slow. That
   single part decides whether the product works."*
5. Hand to Anas.

## Study material — everything explained

**Microsecond.** One millionth of a second. 1,000 microseconds = 1 millisecond.
For scale: sound travels about 34 centimetres in a millisecond.

**Where 146 microseconds comes from.** The reference microphone sits about 7 cm
ahead of the ear. Sound covers 7 cm in roughly 204 microseconds. Subtract the
time for our own speaker's sound to reach the eardrum, and about 146 microseconds
is left for all the electronics.

**Group delay.** How long a chip takes to pass sound through. It exists because
converting sound to numbers requires filtering, and filtering takes time. This is
the single most important hardware number in our project.

**Why predictable noise tolerates delay.** If a sound repeats every 20
milliseconds, then knowing what happened 20 ms ago tells you what is about to
happen. You can be "late" and still be right. Gunfire and wind have no repeating
pattern, so being late means being wrong.

**Coherence — the second limit.** Written **γ² (gamma squared)**, a number from 0
to 1 for how much two microphones hear *the same thing*. The rule is
`NR ≤ −10·log₁₀(1 − γ²)`: at γ² = 0.95 you can never cancel more than 13 dB, no
matter how good your software is. **This is a limit on where you put the
microphones, not on your code.** We measured that coherence matters far more than
speed: improving γ² by 0.03 was worth **+3.3 dB**, while giving the system 4
milliseconds of extra head start was worth **0.3 dB**.

---

# 3 · ANAS — Technical approach (slide 3)

## Your job in one line
Explain the three layers and *why the split exists*, in what/why/how form.

## What / Why / How

**WHAT the architecture is.** Three loops at three speeds.

- **L0** cancels the noise. Plain arithmetic, no AI, 48,000 times a second, must
  finish in under 100 microseconds.
- **L1** cleans the voice going to the radio. A 24,975-parameter neural network,
  125 times a second, 8 milliseconds of delay.
- **L2** understands the scene. Estimates how the noise is *behaving* and writes
  settings into L0 and L1, 62 times a second.

**WHY it is split this way.** Because of the 146-microsecond budget. A neural
network cannot finish in that time on a cheap chip. So we put the intelligence
where the deadline is loose. **L2 gets 16 milliseconds per decision — a hundred
times more time — because nothing waits for it. L0 keeps running on the settings
it last received.**

**HOW L2 helps without touching sound.** We measured it: letting L2 re-tune the
filter as the noise changes is worth **+9.3 dB**. That is the AI contribution,
and it never enters the audio path.

## Pointers you must cover

1. The three layers, their speeds, and which one has AI in it.
2. **The core decision:** *"The teal lines in the diagram carry numbers, never
   sound."*
3. Why L2 predicts **behaviour, not labels** — a label like "helicopter" only
   works for noises we trained on; a vehicle we have never heard still has
   measurable repetitiveness, so the settings still come out right.
4. **The safety guarantee:** every mask value is between 0 and 1, so the output
   can only ever be a *quieter* version of what the microphone heard. The model
   **cannot invent words.** We tested it: zero of 77 million frequency slices
   exceeded the input.
5. If asked about phase: see the "difficult question" section at the end.

## Study material — everything explained

**Mask.** The neural network does not output audio. It outputs a *mask*: for each
slice of sound, a number between 0 and 1 meaning "keep this much". 0 deletes,
1 keeps untouched. Because the number can never exceed 1, the output can only be
quieter than the input — that is our anti-hallucination guarantee, and it is
structural, not a promise.

**Spectrogram / STFT.** Short-Time Fourier Transform. We chop the sound into
32-millisecond slices and, for each slice, work out how much energy is at each
frequency. The result is a picture: time across, frequency up. The network looks
at that picture.

**ERB bands.** We do not handle all 257 frequency slices separately — we group
them into 48 bands, spaced the way human hearing works: fine detail at low
frequencies, coarse at high. Fewer numbers to predict means a much smaller model.

**FxNLMS** — Filtered-x Normalised Least Mean Squares, what L0 runs. In plain
words: it keeps a list of numbers (a filter), multiplies recent microphone
history by it to make anti-noise, listens to the leftover noise, and nudges the
filter to make the leftover smaller. "Filtered-x" means it accounts for the
speaker and air changing the sound on the way to the ear. "Normalised" means it
divides by how loud the reference is — which we found is *itself* a defence
against gunshots, because a huge bang gets divided back down instead of blowing
the filter up.

**Parameters.** The numbers a neural network learns. Ours has 24,975. For
comparison, GTCRN — the published efficiency benchmark we measured ourselves
against — has 48,200. We are half its size.

**Reference mic vs error mic.** The reference microphone is *outside* the cup: it
hears the noise coming so we can predict it. The error microphone is *inside*: it
hears what is left after cancelling, which is the signal that tells the filter how
well it is doing.

---

# 4 · SURYANSH — Feasibility and Viability (slide 4)

## Your job in one line
Prove it can actually be built, and show we found the problems before the judges
did.

## Slide format — two tables, like the reference deck

### Left table — FIELD / SUMMARY

| Field | Summary |
|---|---|
| **Technical** | Runs on a ₹409 chip. Full timing budget **PASSES at 65% usage** |
| **Financial** | **₹6,370** verified bill of materials, everything ships within India |
| **Market** | Armoured vehicle and helicopter crews; also aviation and heavy industry |
| **Operational** | Retrofits a standard ear defender. No new training to use it |

### Right table — CHALLENGE / OUR STRATEGY

| Challenge | Our strategy |
|---|---|
| **Ordinary audio chips are 4× too slow** | Chose the part from a measured latency budget: 38 µs, not 619 µs |
| **Gunshots can destabilise the filter** | Normalisation plus a robust update rule — a loud bang gets divided back down |
| **Cheap microphones hide the same delay inside them** | Analog microphones only on the cancelling path |
| **The chip cannot run a big filter at full speed** | Measured the shortest filter that still works: **L = 128** |
| **A long part lead time could kill the build** | 41-week lead time, so we buy three units, not one |
| **Strong sealing may reduce microphone agreement** | Logged as a risk to measure on the first board, not assumed away |

## Pointers you must cover

1. **Everything fits.** Full budget: L0 172, L1 59, L2 22, audio I/O 6 → **259 of
   400 MMAC/s, 65%, deadline PASSES.**
2. **We priced it properly.** ₹6,370 verified against live vendor pages, not
   estimated. Zero import shipping.
3. **The honest trade:** the short filter costs **7.5 dB on one specific noise
   type** — a 200 Hz tonal rotor component — and nothing on the other three we
   tested. Say it out loud. *"L = 128 satisfies real-time compute constraints
   with negligible loss for broadband and engine conditions, while a 200 Hz tonal
   rotor component benefits from longer memory."*
4. **One catch worth telling them**, because it shows real engineering: the cheap
   digital microphones every hobby shop stocks contain their own converter — the
   exact delay we rejected the ordinary chip for, hidden inside the microphone.
   We found that and designed around it.

## Study material — everything explained

**MMAC/s.** Millions of multiply-accumulate operations per second — the unit of
"how much maths per second". A multiply-accumulate is one multiply plus one add,
the basic operation of all signal processing and neural networks. Our chip does
about 200 million per second per core, and it has two cores.

**Why the filter length matters.** L0's filter is a list of L numbers. More
numbers means it can "remember" further back, which is what lets it cancel a
repeating sound. But more numbers means more maths per sample. At 192,000 samples
per second, L = 128 fits on one core and leaves the other free; L = 512 does not
fit at all. So L is where compute meets acoustics.

**Why 200 Hz specifically loses.** A 200 Hz tone repeats every 960 samples, so a
512-long filter spans over half a cycle and can exploit the repetition. A
128-long filter cannot. A 50 Hz tone needs 3,840 samples, so *no* tested length
reaches it and they all perform the same — which is why 50 Hz shows no loss.

**Lead time.** How long a manufacturer takes to make more chips once stock runs
out. Ours is 41 weeks — about ten months. Our deadline is 20 September. Losing
our only chip while hand-soldering would idle the project for most of a year,
which is why we buy three.

**If asked "why not a Jetson?"** The problem statement says Jetson *"or similar
platforms"*. A ₹50,000 Jetson makes the machine learning easy and the product
unshippable. We showed a ₹409 chip is enough by measuring the budget rather than
assuming it. That is a stronger result, not a compromise.

---

# 5 · RAJ — Impact and Benefits (slide 5)

## Your job in one line
Say who this helps, how much, and be straight about where we currently stand.

## What / Why / How

**WHAT the impact is.** Clearer orders in armoured vehicles and helicopters.
Fewer repeated transmissions. Less hearing damage over a career — military noise
standards changed in 2015 precisely because the old loudness-based rule was not
protecting people.

**WHY it matters beyond defence.** The same problem exists in aviation ground
crews, mining, and heavy industry. The defence version is the hardest case; if it
works there it works in the easier ones.

**HOW we know it works.** Measured on 300 clips of real defence noise the model
has never heard, from recordings kept strictly separate from training.

## Pointers you must cover

1. **The results table, honestly:**

| | noisy input | RHEAR | target | our ceiling |
|---|---|---|---|---|
| STOI | 0.769 | **0.814** | 0.85 | 0.988 |
| PESQ | 1.29 | **1.63** | 2.5 | 4.11 |
| noise removed (SI-SIR) | — | **16.4 dB** | 15 dB | 37.8 |
| net (SI-SDR) | 2.5 dB | **10.1 dB** | 15 dB | 22.7 |

**Then immediately give the breakdown, because it is the stronger number:**

| input SNR | STOI | vs 0.85 target |
|---|---|---|
| below 0 dB (a third of the set) | 0.62–0.70 | fails |
| 5–10 dB | **0.900** | **passes** |
| 10–15 dB | **0.919** | **passes** |
| above 15 dB | **0.958** | **passes** |

*"We meet the intelligibility target at every signal-to-noise ratio at or above
5 dB. Our average is pulled under it by the third of our test set we deliberately
put below 0 dB — where the noise is louder than the speech. We built the harder
test on purpose, and we report the average it produces."*

2. **Say we are short, then say why that is fine.** *"We are below target on two
   of three. But we measured the ceiling of our own design: 0.988 STOI. The
   design is not the limit — the training is. That makes this a schedule problem,
   not a redesign."*
3. **The finding to be proud of.** We remove **16.4 dB of noise** but only net
   **10.1**, because distortion caps us at 11.9. **We do not have a noise
   problem. We have a speech-damage problem.** Most teams never see this, because
   it only shows up if you split the score into "noise removed" and "damage
   caused" instead of reporting one number.

   **On the 15 dB requirement, quote both numbers and say which is which.**
   SI-SIR — noise actually removed — is 16.4 dB and clears it. SI-SDR — the net
   figure after our own distortion — is 10.1 dB and does not. Say both. Picking
   the flattering one is the fastest way to lose a panel that knows the
   difference, and this one will.
4. **We killed four of our own ideas** with measurements: a harmonic filter, mask
   post-processing, pruning, and our own direction hypothesis. Say so — a team
   that publishes its failures is more credible than one with only wins.
5. Move to references without a pause.

## Study material — everything explained

**SI-SIR and SI-SAR.** The usual score, SI-SDR, mixes two different things:
noise you removed, and damage you caused. Splitting it gives **SI-SIR** (how much
noise was removed — higher is better) and **SI-SAR** (how much damage you caused
— higher means less damage). The low one is what limits
us. Ours are 16.4 and 11.9. Reporting only SI-SDR hides this completely, and
finding it changed our entire plan.

**Ceiling / oracle.** We hand the system a *perfect* answer — the exact mask a
cheating model would produce if it already knew the clean speech — and measure
the score. That is the best our design could ever do. Ours is 0.988 STOI and 4.11
PESQ, both far above target. If the ceiling had been low, the design would be
wrong and we would have to start over. It is not, so we keep going.

**Held-out / unseen data.** The model is tested on noise recordings it never saw
in training. We verify this with a script that checks every source, and it has
caught real mistakes. Without that check, good numbers mean nothing.

**MIL-STD-1474E.** The US military noise standard. In 2015 it dropped the simple
"peak loudness" rule in favour of a model of how the ear actually gets damaged —
evidence that impulse noise is a live, unsolved problem, not a settled one.

---

# 6 · RAJ — Research and References (slide 6, 40 seconds)

Do not read the list. Name three or four and say what each *gave* us.

| Work | What it gave RHEAR |
|---|---|
| **Kuo & Morgan**, *ANC: A Tutorial Review* | The foundation — FxLMS and the causality condition everything sits on |
| **SFANC** (2021), **GFANC** (2023), **PFANC** (2026) | A network *selects* a filter per frame instead of adapting sample by sample. This is the seed of our L2 |
| **D-SFANC / PD-SFANC** (2026) | Directional filter selection — our closest prior work, and the source of a limitation we designed around |
| **Deep ANC** (Zhang & Wang, 2021) | Beats classic methods by >6 dB. **The strongest argument against us**, so we must be able to answer it |
| **GTCRN** (2024), **LiSenNet** (2025) | 48.2k and 37k parameters — the efficiency benchmark we measured ourselves against |
| **DCCRN** (2020), **MP-SENet** (2023) | Phase is not a leftover detail. Ironically our own measurement showed our band-level phase correction *hurt* |
| **Wang, Zhu & Kodrasi** (2021) | Speech-presence probability as a second training task — free regularisation that protects weak consonants. We use it |
| **PercepNet** (2020) | Harmonic reconstruction at tiny cost. We tested it, it did not help us — and that null result is what found our real problem |
| **MIL-STD-1474E** (2015) | The military standard moved off peak loudness — evidence the impulse problem is live |

## What to say — 40 seconds

*"Roughly twenty sources. Three shaped the design. Kuo and Morgan gave us the
causality condition our whole architecture obeys. The SFANC family showed a
network can select a filter rather than sit in the audio path — that became our
L2. And Deep ANC is the strongest argument against our approach: it beats
classical methods by six decibels. Our answer is that it has to run inside 146
microseconds on a chip costing a few hundred rupees, and it cannot. So we put the
intelligence where the deadline is loose."*

## The honest bit, if there is time

*"One of these we tested and rejected. PercepNet's harmonic filter should have
improved quality. It gave us four thousandths of a PESQ point — nothing. That
null result is what told us our problem was distortion, not leftover noise, and
it redirected the whole project."*

---

# WHAT EVERYONE MUST KNOW

Learn these five. Judges pick one person and push.

1. **The one-line pitch.** *"A headset where a small AI reads the noise
   environment sixty times a second and re-tunes a conventional canceller — so
   the AI never has to be fast enough to sit in the audio path."*

2. **The causality budget: 146 microseconds.** Anti-noise must beat the noise to
   the ear. This is why no neural network is in the fast loop and why the chip
   choice decides the product.

3. **The three scores.** STOI = understandable (target 0.85). PESQ = natural
   (target 2.5). SNR = speech above noise (target 15 dB).

4. **Where we stand, with the ceiling.** 0.814 / 1.63 / 16.4 dB noise removed.
   Below target on two. **Never quote the score without the ceiling of 0.988** —
   the ceiling is what makes this a schedule problem rather than a design flaw.

5. **We remove 16.4 dB of noise and net 10.1, because damage caps us at 11.9.**
   Our problem is speech damage, not noise. Volunteer this; it is our sharpest
   finding.

## Six questions to rehearse

**"ANC headsets already exist — Bose, 3M Peltor."**
Those cancel steady noise. A normal audio chip's 619 µs delay gives **zero dB**
on wideband noise — gunfire, wind, blast. Steady hum survives long delays because
it repeats; unpredictable noise does not.

**"Why not a bigger neural network? Deep ANC beats classical by 6 dB."**
It does, and we cite it. But a network in the audio path must finish inside
146 µs on a chip costing a few hundred rupees — that is a handful of operations,
not a network. We measured it. So we put the intelligence where the deadline is
loose.

**"Where is the AI, then? This sounds like ordinary signal processing."**
Two places. L1 is a real neural speech enhancer on the outgoing voice. L2
estimates the acoustic *state* and converts it into filter settings. The AI
carries numbers into the loops, never sound through them.

**"What is your accuracy?"**
Refuse the single number. *"STOI 0.814 against 0.769 for the raw input, 16.4 dB
of noise removed, on 300 clips of real defence noise with verified zero overlap
with training. Target is 0.85 and our architecture's ceiling is 0.988."*

**"Have you built the hardware?"**
No, and say so immediately. Everything is measured in a simulator validated
against published results. The parts list is specified to part numbers with a
latency justification. That sequencing is deliberate — the simulator already
rejected one chip and one processor before we spent money.

**"The problem statement asks for phase preservation. You removed it."**
This is the hardest question. *"Our input is still complex — real, imaginary and
magnitude. What we removed is the band-level phase rotation, because it is
computed for 48 bands and then applied to all 257 frequency slices, and our
widest band covers 19 of them. Phase wraps faster than that, so one angle cannot
be right for a whole band. We measured it: removing it improved understandability,
naturalness and distortion on 600 clips. And our ceiling test shows a perfect
magnitude mask with the phase left alone already reaches PESQ 2.84 — above the
target. So phase prediction is not required to hit the numbers."*

## Three things nobody should say

- **"Crystal clear"** or any absolute quality claim. We have numbers; use them.
- **"It works"** without naming the test set.
- **Guessing.** If you do not know, say *"we measured that, I'll get you the
  exact figure"* and hand to your backup. Every number in this deck came from
  code that ran — nobody should invent one on stage.

---
---

# PART II · MARKET, VIABILITY AND BUSINESS CASE

*Answers the mentor's four questions and the team lead's four claims. Every
figure sourced; market forecasts are third-party projections and labelled so.
RHEAR's own numbers come from code run against frozen, checksummed test sets.*

**Visual version for the slide deck:** https://claude.ai/code/artifact/c80553e9-edcb-413c-bfe8-ba1d85c72a99

---

## A · The problem is real, and already expensive

| Evidence | Figure |
|---|---|
| Noise-induced hearing loss, post-deployment records | **21%** |
| Tinnitus, same records | **28%** |
| Combat arms soldiers with mild loss or worse (earlier study) | 30% |
| Rank among all service-connected disabilities | **2nd most prevalent** |
| US Veterans Affairs compensation for hearing damage | **over $1 billion / year** |

**MIL-STD-1474E (2015)** dropped its simple peak-loudness rule for a model of how
the ear is actually damaged — an official admission that the impulse-noise
problem was *not* solved.

**Say it like this:** *"This is not a comfort feature. Hearing damage is the
second most common disability among service personnel, and one government alone
pays over a billion dollars a year for it."*

---

## B · Who the customers are

| Customer | Why they need it | Route |
|---|---|---|
| **Indian Army** — armoured & mechanised crews | Tank/ICV interiors are enclosed, low-frequency, continuous. Crews already wear tethered intercom headsets | iDEX / DRDO → MoD |
| **IAF & Army Aviation** | Rotor noise is the hardest broadband case; comms failure is a flight-safety issue | Same |
| **Artillery & infantry** | Impulsive blast — the case classical ANC handles worst | Same |
| **Paramilitary & police tactical** | CRPF, BSF, NSG — same acoustics, larger headcount, simpler procurement | MHA |
| **Civil aviation ground crew** | Continuous jet noise, mandated protection, must still hear instructions | Commercial |
| **Mining, steel, heavy industry** | Legally mandated hearing protection that currently blocks speech too | Commercial |

**Beachhead: armoured vehicle crews.** They *already* wear a headset tethered to
the vehicle intercom, so a wired reference sensor and a connector are normal
rather than a new burden.

---

## C · Market size

| Market | Today | Forecast | CAGR |
|---|---|---|---|
| Global tactical communication headsets | **$3.64 B** (2026) | **$5.81 B** (2033) | 7% |
| India defence electronics | **$6.85 B** (2024) | **$11.35 B** (2032) | 6.2% |

The first is the product category. The second is the budget we would actually be
procured from. **We do not need market share to matter — we need one programme.**

---

## D · The budget line already exists — the strongest India argument

India MoD, FY 2025–26:

| Stage | Amount |
|---|---|
| Total MoD budget | ₹6,81,000 cr |
| Capital outlay | ₹1,80,000 cr |
| Capital acquisition (modernisation) | ₹1,48,723 cr |
| **Reserved for DOMESTIC sources — 75%** | **₹1,11,545 cr** |
| **Domestic PRIVATE industry carve-out** | **₹27,886 cr** |

MoD **fully utilised** its capital outlay in FY26 — the money is being spent, not
lapsing.

**Say it like this:** *"Three quarters of India's modernisation budget is
ring-fenced for domestic sources, and ₹27,886 crore of it specifically for
domestic private industry. A low-cost India-built headset is not fighting policy
— it is exactly what the policy was written to fund."*

---

## E · Who is already in this market

| Player | What they sell | Where the gap is |
|---|---|---|
| **3M / Peltor** | Category leader. Passive defenders and comms headsets, widely available in India | Designed for **steady** noise. No published adaptive retuning for non-stationary or impulsive noise |
| **INVISIO** | High-end tactical comms and hearing protection | Imported, premium, procured as a system rather than integrated into an existing vehicle headset |
| **Silynx, Safariland, David Clark, Otto** | Tactical/aviation headsets with DSP and ANC | Same family — classical ANC tuned for continuous noise |
| **Bose** | Aviation ANC, excellent at steady cabin noise | Consumer/aviation lineage; not built around impulsive blast |
| **Indian PSUs** (BEL and similar) | Communication equipment for the forces | Integration and manufacture rather than novel adaptive-ANC research |

**State this carefully.** These are capable companies and all of them do noise
cancellation. **We are not claiming they cannot suppress noise.** The gap we
target is specific: noise that changes every few seconds, and 160 dB impulses.

---

## F · How we are different

| | Typical existing product | RHEAR |
|---|---|---|
| Noise assumption | Steady, continuous | **Changes every few seconds** |
| Impulsive blast | Passive attenuation only | Robust update rule; normalisation as a blast defence |
| Adaptation | Fixed or slowly adaptive filter | **AI rewrites filter settings 62×/second** |
| Where the AI sits | — | **Outside the audio path** — so it never needs to be fast |
| Part selection | Datasheet-driven | Chosen from a measured **146 µs** latency budget |
| Bill of materials | Imported system | **₹6,370**, ships within India |
| Can it invent words? | — | **Structurally impossible** — tested on 77M samples |

**The measurement that proves the point:** on unpredictable noise — gunfire, wind,
blast — cancellation falls from **−13.7 dB at 38 µs** to **exactly −0.0 dB at
619 µs**, the delay of an ordinary audio chip. Steady hum survives long delays
because it repeats; impulsive noise cannot be predicted, so being late means
being wrong.

---

## G · Cost — the number that decides adoption

| | |
|---|---|
| Our verified bill of materials | **₹6,370** |
| Our processor | **₹409** |
| The edge-AI board the PS names as an example | **~₹50,000** |
| Import shipping | **₹0** — every part sources domestically |

The PS names an NVIDIA Jetson *"or similar platforms"*. A ₹50,000 board makes the
ML easy and the product unshippable at scale. We showed a ₹409 chip is enough by
**measuring** the compute budget — 259 of 400 MMAC/s, frame deadline passes at
65% usage. **That is a stronger result than using the expensive part, not a
compromise.**

---

## H · Government support this fits

| Policy | How RHEAR fits |
|---|---|
| **iDEX** | The route this PS already runs through — built for startups and students solving defence problems |
| **75% domestic procurement mandate** | ₹1,11,545 cr reserved for domestic sources. We are domestic by construction |
| **₹27,886 cr private-industry carve-out** | A new private venture is directly eligible |
| **Make in India / Atmanirbhar Bharat** | Import substitution in a category dominated by foreign suppliers |
| **Positive Indigenisation Lists** | The mechanism closing imported items to imports over time — the direction of travel favours a domestic option existing |

---

## I · Why this team can build it — evidence, not promises

| Claim | Evidence |
|---|---|
| The design is physically possible | 146 µs causality budget derived; chip chosen from it. **−13.7 dB vs −0.0 dB** measured |
| The AI contributes | **+9.3 dB** from L2 retuning the filter as noise changes |
| It fits the chip | Full timing budget **263 of 400 MMAC/s**, 66%, frame deadline **PASS** |
| It can be bought | **₹6,370** verified to part numbers, all domestic |
| The design is not the limit | Measured ceiling **STOI 0.988** against a 0.85 target |
| We test honestly | Two held-out sets, verified zero training overlap. **Four of our own ideas rejected on measurements** |

**Where we stand today, stated plainly:** STOI 0.814 against 0.85 and PESQ 1.63
against 2.5 — short on both. We remove **16.4 dB** of noise but net 10.1,
because distortion caps us at 11.9. That is a speech-damage problem, not a noise
problem. **No hardware is built yet.**

**And we know precisely why each one is short, which is not the same as being
stuck:**

- **STOI is a distribution story.** We already exceed 0.85 at every input SNR at
  or above 5 dB (0.900 / 0.919 / 0.958). The average is pulled under by the third
  of our test set we deliberately placed below 0 dB.
- **PESQ is a design story, and we found it by measuring our own ceiling.** A
  *perfect* mask on our 48-band grid, leaving the noisy phase alone, scores only
  **2.838**. The target is 2.5. Reaching it would demand 88% of a perfect
  model — which no 25,000-parameter network will deliver. **More training cannot
  close this gap; the second stage described below is what closes it.**

Knowing which of the two is a schedule problem and which is a design problem is
worth more than either number.

---

## J · The one change that closes the PESQ gap — and why we know it works

**Anyone may be asked this. Manaswi and Anas must be able to give it in full.**

### The problem in one sentence

Our model outputs **one real number per frequency band** — a volume knob per
band, 48 of them. A knob can turn a band down. It cannot repair the fine
structure inside that band, and it cannot touch phase. That is why our perfect-mask
ceiling is PESQ 2.838 and not 4.5.

### What everyone who solved this did

The two most-cited low-complexity speech enhancers in the field are built on
**exactly our first stage** and then add a **second stage** we do not have:

| | bands | second stage | cost |
|---|---|---|---|
| **PercepNet** (Valin, Interspeech 2020) | 34 ERB | comb filter on the pitch | < 5% of a CPU core |
| **DeepFilterNet 1/2** (Schröter, ICASSP 2022) | **32 ERB** | **deep filtering** | runs on a Raspberry Pi 4 |
| **RHEAR today** | **48 ERB** | **none** | 62.7 MMAC/s |

Read the band column first: **both use FEWER bands than we do.** Our resolution
was never the problem, which is exactly what our own ceiling probe found. The
difference is the second stage.

DeepFilterNet's published ablation puts a number on it: the deep-filtering stage
adds roughly **+0.2 PESQ and +2 to +2.8 dB SI-SDR over ERB gains alone.**

### What deep filtering actually is

Instead of one real gain per band, the network predicts a **short complex filter
across time** for each low-frequency bin — five taps — and applies

> Y(t, f) = Σ over i of C(i, f) · X(t − i, f)

Because the coefficients are complex and span several frames, this stage *can*
correct phase and reconstruct periodic structure. It is therefore **not bounded
by the 2.838 ceiling that limits our real-valued mask.** That is the whole point:
it does not push us closer to our ceiling, it raises the ceiling.

### What it costs us — measured, not hoped

Applied below 5 kHz (160 of our 257 bins) at order 5:

| | |
|---|---|
| applying the filter | 0.80 MMAC/s |
| the extra decoder that emits the coefficients | ~12.8 MMAC/s |
| **total added** | **~13.6 MMAC/s** |
| core 1 goes from | 84.8 → **98.4 of 200** |
| frame deadline | **still PASS** |

Under 14 MMAC/s on a budget with 115 spare. **We have the compute; we were
simply missing the stage.** (Our frame rate is 250/s — a 4 ms hop — which is
what these figures are computed at.)

### How to say it if a judge presses on why you are below target

*"We are below target on PESQ, and we can tell you exactly why rather than
promising more training. We measured the ceiling of our own representation: a
perfect mask on our band grid scores 2.838, so the target sits at 88% of perfect
— unreachable by training a 25,000-parameter model harder. The fix is structural
and it is known: the two most-cited low-complexity systems in this field,
PercepNet and DeepFilterNet, use fewer bands than we do and add a second stage
for the fine structure. DeepFilterNet's own ablation puts that stage at about
+0.2 PESQ. We costed it for our chip: under 14 MMAC/s, and the frame deadline
still passes. That is our next build, and it is scheduled, not speculative."*

**Do not oversell it.** +0.2 PESQ does not by itself take 1.63 to 2.5. Say what
is true: it raises a ceiling that currently makes the target unreachable, and it
is the necessary first step rather than the whole answer. A panel will respect
"we know the mechanism and we costed it" far more than a promise of a bigger
number.

**References for this section:** Valin et al., *A Perceptually-Motivated
Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech*,
Interspeech 2020 (arXiv 2008.04259) · Schröter et al., *DeepFilterNet*, ICASSP
2022 (arXiv 2110.05588) and *DeepFilterNet2*, IWAENC 2022 (arXiv 2205.05474).

---

## J · Sources

- India MoD budget FY2025–26 — PIB and MP-IDSA
- India defence electronics market — Fortune Business Insights
- Tactical communication headset market — Verified Market Reports, Fortune Business Insights
- Hearing loss prevalence — NCBI / Military Medicine, National Academies
- MIL-STD-1474E (2015) — US Department of Defense
- RHEAR's own figures — code run against frozen, checksummed test sets

*Market forecasts are third-party projections. Say "projected to" when quoting
them, not "is".*
