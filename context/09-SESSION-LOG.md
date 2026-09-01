# How we got here — the arc, and the mistakes

Useful because several conclusions only make sense with the path that produced
them, and because the mistakes are the most transferable part.

## The finding that redirected the project

Splitting SI-SDR into **SI-SIR (noise removed)** and **SI-SAR (damage caused)**
showed ~20 dB SIR against ~10.5 dB SAR. **RHEAR never had a noise problem; it
has a speech-damage problem.** A single SI-SDR number hides this completely.
Everything after this point targets SAR and PESQ, not suppression.

## Then: the ceiling probes

An oracle mask on our own 48-band grid reaches SI-SAR 22.91 dB against our 10.5.
**A training problem, not a representation problem.** Later the full oracle
ladder refined this: our PESQ ceiling is 2.731 against a 2.5 target, so the
target needs 92% of a perfect model — and we achieve 60%.

## Then: the cross-benchmark, which reframed everything

Running GTCRN's published weights on our clips showed a model **twice our size**
scoring **1.643 where we score 1.642**. Most of the gap we had been attacking
was never ours. **This measurement should have come first.** It costs three
minutes and it invalidated the premise of four experiments.

**Transferable lesson: check your target against an external reference before
optimising toward it.**

## Mistakes worth not repeating

**Analysis errors caught by assertions or sanity checks**
- Oracle deep filter solved **one filter per clip** and scored below a real
  mask, which is impossible — a real mask is the special case `c=[g,0,0,0,0]`.
  It was measuring time-invariant vs time-varying.
- The same solve had the **conjugate on the wrong index** (`R` transposed),
  solving `conj(R)c = p`. Fixed, then unit-tested: 5 taps must beat 1 tap.
- Deep filtering was first modelled as a **replacement** for the ERB mask below
  5 kHz. It is a **refinement on top**. Fixing that moved the number 2.275→3.297.

**Process errors that cost hours**
- Left the demo twin running at 70% CPU during a training run, for two hours.
- Used `--cache-int16` on a 16 GB laptop: 5 GB cache paged out, epochs went
  26 → 29 → 50+ min. **A run getting monotonically slower is a resource problem.**
- Piped training through `tail`, so no progress was readable; then inferred
  epoch times from file timestamps and **got three estimates wrong in a row**.
- Printed every 5th epoch, hiding epochs 1–3 of a 5-epoch run behind an empty log.
- Renamed metric keys in `summary.json`, breaking the dashboard's A/B panel so
  it looked like the audio had been deleted.
- A blanket `runs/` in `.gitignore` silently excluded the demo's own audio; the
  pushed repo served the dashboard and 404'd every clip. **Verify by cloning
  from the remote, not from the directory you committed from.**

**A false alarm worth knowing about**
- A leakage alarm on the H3 dataset was **my error**, not a real leak: the
  exclusion set included throwaway 1-clip train/val rows. The test split was
  always clean. It nearly triggered a pointless rebuild.

## What the user has been consistently right about

- "Don't hide the 7.5 dB rotor penalty" — say the cost, then the mitigation, and
  never claim L2 "solves" it.
- "Make sure the full RHEAR timing budget is recorded, not only L0."
- Insisting on Indian data and sources over US figures for the market case.
- Calling a stop after four hours of marginal results and demanding research
  first. That produced the cross-benchmark, which is the best result we have.
- The G7 plan (keep the architecture, move the operating point, spend the freed
  compute on capacity, don't add MetricGAN yet) — this is sound and evidence-led.

## Style notes for whoever picks this up

The user wants: real measurements over claims, negative results recorded, one
variable at a time, no overselling, and **the cost stated alongside the benefit**.
They will catch an unstated caveat. They prefer being told "this is a weak
negative and here is why" over a clean-sounding verdict that hides its dose.
