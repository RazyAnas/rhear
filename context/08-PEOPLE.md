# People, presentation, logistics

## Team — six members, 10-minute pitch, six slides

| slide | member | topic |
|---|---|---|
| 1 | **Manaswi** | Idea / solution (opens) |
| 2 | **Faraz** (with Manaswi) | the physics that forces the design |
| 3 | **Anas** | technical approach |
| 4 | **Suryansh** | feasibility and viability |
| 5 | **Raj** | impact and benefits |
| 6 | **Raj** | research and references (40 s) |

Work split as given: Faraz / Raj Kamal / Vishnu Raj → hardware + presentation.
Suryansh / Anas / Vishnu Raj → software + presentation. Manu / Faraz →
presentation lead.

Full scripts with study material: **`docs/PRESENTATION-SCRIPTS.md` (+ PDF, 17
pages)**. It includes every term explained, anticipated questions, and a
Section J on the deep-filtering roadmap.

## The rules for presenting numbers

1. **Never say a number without saying what it is measured on.**
2. **Give the SNR breakdown, not just the average.** We pass STOI at every SNR
   >= 5 dB; the average is dragged under by the third of the set below 0 dB.
3. **On "SNR > 15 dB", quote both and say which is which.** SI-SIR 16.4 clears
   it; SI-SDR 10.05 does not. Picking the flattering one loses the room.
4. **Volunteer what is not built.** No hardware exists. The simulator rejected
   two components before money was spent — that is the argument.
5. **Cite the cross-benchmark.** A published model twice our size scores 1.643
   on our data. That reframes 1.63 from failure to parity.

## Who has the GPU

**Suryansh** runs the long training jobs. Handoff materials in
`~/PS#2/handoff-g12/` (code only, ~100 KB) and `~/PS#2/handoff/` (2.7 GB with
data). `handoff-g12/README.md` has exact commands and sanity checks.

**On his box `--cache-int16` is correct.** On the laptop it is not (gotcha 14).

History: his first run was the wrong variant (22,988 params = phase branch
present). The README now has a 30-second check table. G1 was later confirmed to
have run correctly — its epoch-0 loss of +11.17 is the perceptual signature.

## Deadline

**20 September 2026.** The codec's 41-week manufacturer lead time means if
current stock clears, the next batch lands after the deadline.
