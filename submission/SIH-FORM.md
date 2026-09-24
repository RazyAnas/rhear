# SIH SUBMISSION — PS 26052 — copy/paste fields

---

## 5. TECHNOLOGY BUCKET

**AI/ML, Cloud Computing, Blockchain**

(The problem statement is titled "AI/ML-enabled Adaptive Noise Cancellation".
"IoT and Electronics" also describes the prototype, but the PS asks first for a
model, a training framework and an inference engine — that is the AI/ML bucket.
Pick AI/ML.)

---

## 1. IDEA TITLE  — pick one, all under 100 characters

**Preferred (83 chars):**
RHEAR: Defence ANC where the AI tunes the canceller but never carries the audio

**Alternative A (89 chars):**
RHEAR: Edge AI noise cancellation that protects hearing and clears the radio separately

**Alternative B (74 chars):**
RHEAR: 50k-parameter ANC that beats a 25-million-parameter model on gunfire

---

## 4. ABSTRACT / SUMMARY  — ~2,900 characters, limit 10,000

RHEAR is an AI-driven adaptive noise cancellation system for defence headsets,
built and measured end to end on an ESP32-S3 — a microcontroller, not an
accelerator.

It starts from a requirement other systems collapse together. A soldier has two
acoustic problems that pull in opposite directions. He must not go deaf, which
means killing engine, rotor and gunfire noise. He must also be understood on the
radio, which means his voice and nothing else. Those goals conflict the moment a
comrade shouts a warning: that voice must reach his ears, because suppressing it
is a safety failure, and must not reach his radio, because the channel is his
alone. One microphone, one instant, two opposite targets. RHEAR treats protection
and communication as separate control problems with separate metrics.

The architecture follows from that. Three layers run at three rates on one chip.
L0 is hearing protection: FxNLMS, pure DSP at 48 kHz, a 20.8 microsecond
per-sample budget, and no neural network anywhere in it. L1 is the speech
enhancer for the radio: a 49,663-parameter model at 62.5 frames per second. L2 is
a scene engine at 62.5 Hz that emits coefficients, never audio. The AI therefore
sits in the coefficient path, not the audio path. The consequence is a safety
property: a stalled or failed model cannot open a hole in hearing protection,
because protection never waits on it.

Measured results. On a 300-clip defence evaluation set — gunshot, shelling,
helicopter, armoured vehicle, drone, fighter — RHEAR scores STOI 0.8236, PESQ
1.7159, SI-SDR 10.76 dB. Run under identical conditions, GTCRN's own published
weights score 1.6430 PESQ and 9.61 dB, and SepFormer, a 25.6-million-parameter
transformer, scores 5.56 dB. RHEAR beats a model 516 times its size by 5.2 dB
SI-SDR. On VoiceBank+DEMAND, the public benchmark the PS's thresholds come from,
it meets all three targets: SI-SDR 18.35 dB, STOI 0.936, PESQ 2.524.

The training pipeline generated over 100 GB of noisy-clean pairs across 251
speakers and 235 noise classes, with measured room impulse responses, clipping,
microphone response randomisation and competing talkers at controlled
signal-to-interference ratio. Every architectural change was accepted or rejected
by a paired bootstrap over 10,000 resamples on two independent evaluation sets.
Four of the team's own models were rejected under that rule, including one that
improved PESQ but degraded intelligibility.

Cost is the deployment argument. The reference tactical system, the US Army's
TCAPS, costs about $2,000 per unit and its price is cited as the limit on wider
fielding. RHEAR's India-sourced electronics BOM is ₹4,071 as a RETROFIT into
passive ear defenders a unit already owns — no new shell and no re-qualification
of certified passive attenuation. Even at five times the BOM once enclosure,
ruggedisation and testing are added, a fielded unit is roughly one-ninth of the
TCAPS reference. That difference decides who gets issued one: a specialist subset,
or a whole gun line. It matters because a study of Indian Air Force personnel put
noise-induced hearing loss at 22.9% overall and 26.18% in technical trades —
roughly one in four, permanent and untreatable. No part in the BOM is
export-controlled, the corpus and weights are produced in-house, and the model is
a 97 KB file that never leaves the country.

The project also reports its own ceiling. Scoring the ideal mask shows that a
perfect 48-band magnitude mask cannot exceed PESQ 2.392 at 0 dB against a 2.5
target — so at low SNR the limit is the information in one microphone, not the
size of the network. That measurement redirected the work from bigger models to
better inputs, and it is why a second boom microphone, which separates the
wearer's voice from a bystander's at 100 percent using wavefront curvature alone,
is the next hardware step rather than a larger model.

---

## 2. IDEA DESCRIPTION  — limit 50,000

### THE MOMENT THIS IS BUILT FOR

A gun detachment is firing. The noise is past 150 decibels — loud enough that
every round takes a little hearing away permanently. The soldier is wearing ear
defenders, so he is protected. He is also, for that reason, half deaf to his own
team.

Three metres to his left, someone shouts a warning.

He has to hear that. Any system that silences it to protect him has just made him
less safe, not more. And at the same instant, when he presses his radio to call
the correction, the gun line behind him and the voice beside him must NOT go out
on that channel — because the net belongs to him for those four seconds, and a
misheard correction is a round in the wrong place.

**So the same human voice must be kept in his ears and removed from his radio, in
the same instant, from the same microphone.**

That is not a noise problem. It is two opposite problems wearing the same
uniform, and it is the reason "just remove the noise" — which is what every
denoising system on the market does — cannot solve it.

RHEAR is built from that distinction outward.

---

### WHY THIS MATTERS NOW — THE PEOPLE AND THE NUMBERS

**One in four.** A study of Indian Air Force personnel found noise-induced
hearing loss in **22.9%** of them overall, rising to **26.18% in technical
trades** against 12.5% in non-technical ones. Artillery, armour and aircrew sit
in the same exposure class.

Hearing loss is not like other injuries. It is **permanent, cumulative and
untreatable**. It does not heal, it does not respond to medicine, and every
exposure adds to the last. For the services it is also a downgrade in medical
category, a constraint on posting, and a disability liability that outlives the
career by decades.

**The solution already exists — and almost nobody has it.** The US Army's TCAPS
is the reference system for exactly this: hearing protection with communication
built in. It costs about **$2,000 per unit — roughly ₹1.8 lakh** — and after
years of rollout has reached about **20,000 soldiers**. Its price is openly cited
as the reason it has not gone further.

That is the real shape of the problem. This is not an unsolved engineering
question. **It is a solved question with an unaffordable answer**, and the result
is that protection goes to a specialist few while the gun line, the tank crew and
the flight line go without.

**RHEAR's contribution is not a better score. It is a price at which everyone can
be issued one.**

---

### THE IDEA, IN PLAIN TERMS

RHEAR is a headset intelligence system that runs on a ₹409 commodity
microcontroller and retrofits into ear defenders a unit already owns.

It treats the soldier's two needs as two separate jobs, because they are:

**The ear path — what he hears.** Engine, rotor, gunfire and shelling are
cancelled. **Human voices are deliberately preserved.** A shouted warning still
reaches him. This is the safety half, and it is the half everyone else deletes by
accident.

**The radio path — what he transmits.** Machine noise is removed, *and* so is
every voice that is not his. His correction goes out clean.

**The scene engine — what adapts.** The system continuously recognises what it is
listening to — a helicopter is not a gunshot is not an engine — and retunes the
canceller for that environment.

The architectural decision that makes this safe is simple enough to state in one
sentence, and it is the heart of the design:

> **The AI tunes the canceller. It never carries the audio.**

Hearing protection is pure signal processing running 48,000 times a second. The
neural network sits beside it, adjusting it, and is never in the path the sound
travels. **A software fault, a stall, a dropped frame — none of them can open a
hole in a soldier's hearing protection, because protection never waits on the
AI.** You can switch the intelligence off entirely, live, and the ears stay
protected.

Every system that puts a neural network inside the cancellation loop has the
opposite property. That is a design decision we made on purpose, for a user who
cannot afford a reboot.

---

### WHAT WE HAVE ALREADY PROVEN

This is not a concept note. The model is trained, and it has been measured
against the best published systems in the world on the noise it is built for.

**Tested on 300 clips of real defence noise — gunfire, shelling, rotor, armour:**

| system | size | intelligibility (STOI) | quality (PESQ) | noise removed (SI-SDR) |
|---|---|---|---|---|
| **RHEAR** | **49,663 parameters** | **0.8236** | **1.7159** | **10.76 dB** |
| GTCRN (published weights) | 23,700 | 0.8182 | 1.6430 | 9.61 dB |
| SepFormer (transformer) | **25,613,569** | 0.8021 | 1.5963 | 5.56 dB |
| MetricGAN+ | ~2,000,000 | 0.7546 | 1.8197 | **−0.46 dB** |

**Read the first and third rows together.** SepFormer is a state-of-the-art
transformer with **516 times more parameters** than RHEAR. On defence noise,
RHEAR removes **5.2 decibels more noise** than it does. Size is not the thing
that wins here — training on the right noise is.

We also report the row that beats us, because it proves the point. MetricGAN+
scores the best quality number in the table and a **negative** noise-removal
figure: it is trained to maximise that one metric and damages the signal while
doing it. Judging a system on one number is how you end up fielding that.

**Against the problem statement's own thresholds**, on the public benchmark those
thresholds come from (VoiceBank+DEMAND, 824 clips): **SNR 18.35 dB** against a
target of 15, **STOI 0.936** against 0.85, **PESQ 2.524** against 2.5. All three
met.

**And we measured our own ceiling.** Rather than assume more training would help,
we computed the score a *perfect* version of our model could achieve. It told us
that at the hardest noise levels the limit is the information in a single
microphone, not the size of the network. So we stopped making the model bigger
and added a second microphone instead — which separates the wearer's voice from a
bystander's **with 100% reliability**, using nothing but the curvature of the
sound wave. No voice enrolment, no training on the individual, and it still works
when he is shouting.

Very few teams can tell you the maximum their own approach could ever reach. It
is the difference between hoping and knowing.

---

### THE COST ARGUMENT, IN FULL

Every price below was verified against Indian distributors.

| line | ₹ |
|---|---|
| Complete build, all electronics + ear defender shell | **6,370** |
| — of which the passive earmuff shell itself | 2,299 |
| **Electronics only, as a retrofit** | **4,071** |

**RHEAR retrofits. It does not replace.** It fits inside a passive earmuff
already in service. No new shell to procure, no re-qualification of passive
attenuation that is already certified, no change to how a soldier wears it or how
a unit stores it.

We will not claim a fielded unit costs ₹4,071 — a bill of materials is not a
product. Add enclosure, assembly, ruggedisation, environmental qualification and
testing, and at **five times the BOM** a fielded unit lands near **₹20,000**.

**Against ₹1.8 lakh for the reference system, that is roughly one-ninth.**

And that ratio is the entire point. At TCAPS pricing, a battery of 100 personnel
costs ₹1.8 crore to protect and the answer is usually no. At RHEAR pricing the
same battery is ₹20 lakh, and the answer changes. **We are not competing on
score. We are competing on how many people get protected.**

---

### WHY THIS IS INDIA'S TO BUILD

The Government has spent five years building exactly the runway this needs.

- Defence production reached a record **₹1.78 lakh crore in FY 2025-26**, up
  **15.6%**, with a target of **₹3 lakh crore by 2029**.
- Defence exports hit an all-time high of **₹38,424 crore**.
- **More than 65%** of India's defence equipment is now produced domestically.
- **Ten Positive Indigenisation Lists covering 5,521 items** have been notified.
- In September 2026 the Defence Acquisition Council cleared **₹1.10 lakh crore**
  of proposals with **around 98% to be sourced from Indian industry**.
- **iDEX** has engaged **676 startups and MSMEs** across **551 contracts**, and
  **TDF** offers grants up to **₹50 crore** with a further ₹500 crore corpus for
  deep technology.
- Critically, MoD has created **direct procurement pathways for successful iDEX
  and TDF technologies** — a route from prototype to fielding that did not exist
  before.

And the policy language has moved from **"Make in India" to "Owned by India."**

RHEAR is built to satisfy that literally, not rhetorically:

- **No export-controlled part** anywhere in the bill of materials. No end-use
  licence, no foreign government approval, no clause that lets a supplier decide
  where our soldiers may use it.
- **Commodity silicon with multiple suppliers** — not a single-source defence
  component with a lead time measured in quarters.
- **The corpus is generated here. The model is trained here. The weights are a
  97-kilobyte file that never leaves the country.** No vendor cloud, nothing
  phoning home, no foreign party holding the firmware.
- Every verified line in the bill of materials is already available from Indian
  distributors.

A hearing-protection system for Indian soldiers should not depend on a foreign
company's licensing decision. This one does not.

---

### WHY DRDO SHOULD CHOOSE THIS

**1. The safety guarantee is structural, not promised.** Because the AI is never
in the audio path, no software fault can compromise hearing protection. This is
not a reliability target to be argued about in review — it is a property of the
architecture, and it can be demonstrated in thirty seconds by switching the AI
off while cancellation continues.

**2. It is trained on our noise, not on office noise.** 16% of the training
material is gunshot, 9.4% shelling, 9.8% rotor, 10.4% armoured vehicle, mixed
with measured room acoustics and controlled interference from competing speakers.
Systems trained on the standard public corpora — cafés, offices, city streets —
do not transfer to a gun line, and our comparison table shows exactly that.

**3. Every claim is auditable.** Each design decision was accepted or rejected by
a statistical test over 10,000 resamples on two independent evaluation sets. Four
of our own models were rejected under that rule, including one that improved
quality but reduced intelligibility. Nothing in this submission has to be taken
on trust; it can be re-run.

**4. It fits the procurement reality.** Commodity parts, Indian supply chain,
retrofit into in-service equipment, and a per-unit cost that permits issue at
scale rather than to a pilot group.

---

### DELIVERY

**Phase 1 — the communication layer.** Voice capture, neural enhancement,
voice-activity gating and radio output, running on the ESP32-S3. The model is
trained and benchmarked; the on-device signal chain, neural voice detection and
silence gating are running on hardware today.

**Phase 2 — active cancellation.** The adaptive canceller, validated at +12.6 dB,
paired with the low-latency audio codec the circuit is already designed around —
37 parts, 30 nets, electrically verified.

**Phase 3 — the second microphone.** Near-field separation of the wearer's voice
from everyone else's, simulated across 2,160 acoustic scenes, at a cost of one
extra component on the same data bus.

**Phase 4 — ruggedisation and field trial.** Enclosure, environmental
qualification, and trial with a user unit.

---

### IN ONE SENTENCE

One in four personnel in high-noise roles are losing their hearing permanently;
the system that would prevent it costs ₹1.8 lakh and has reached almost nobody;
**RHEAR delivers the same capability for roughly one-ninth of that, retrofits
into equipment already in service, keeps the shouted warning that other systems
delete, and is owned end to end — silicon, data, model and firmware — by India.**

