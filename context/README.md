# RHEAR — full project context

**Read this file first, then `01-STATE.md`.** Everything here is written for an
assistant picking the project up cold. It records what is true, what is
measured, what was rejected and why, and where the traps are.

Written 2026-09-01, ~02:00 IST. One experiment (**G7**) was still training when
this was written — see `01-STATE.md` § "In flight".

## The one-paragraph version

RHEAR is a defence ANC headset for **SIH 2026 Problem Statement 26052** (DRDO /
iDEX, deadline **20 September 2026**). Three layers: **L0** a deterministic
FxNLMS canceller protecting the wearer's ears, **L1** a ~25k-parameter neural
speech enhancer cleaning the outgoing radio voice, **L2** a 1,764-parameter
neural scene engine that rewrites L0 and L1's coefficients 62.5 times a second
and **never touches audio**. Everything is validated in a software digital twin;
**no hardware has been built**. The PS targets SNR > 15 dB, STOI > 0.85,
PESQ > 2.5.

## The single most important result

We spent weeks reading "PESQ 1.63 against a 2.5 target" as a model failure.
**It is mostly not.** Running GTCRN's published weights (ICASSP 2024, 48,245
params — 1.93x ours, published at **PESQ 2.87** on VoiceBank+DEMAND) over our
own defence clips gives **PESQ 1.643**. We score 1.642 with half the parameters
and better SI-SAR.

**The gap is the test set, not the model.** See `02-RESULTS.md` § Cross-benchmark.
This reframes every conversation about our numbers and it is reproducible in
about three minutes: `python3 E03/crossbench.py --ckpt dns3`.

## Files here

| file | what it holds |
|---|---|
| `01-STATE.md` | where everything stands right now, what is in flight, what to do next |
| `02-RESULTS.md` | every number, every experiment, the oracle ceilings |
| `03-REJECTED.md` | eight things we tried and killed, with the evidence |
| `04-CODEBASE.md` | file map and how to run each thing |
| `05-GOTCHAS.md` | the traps that cost hours. **Read before touching code.** |
| `06-HARDWARE.md` | BOM, chips, timing budget, causality |
| `07-EXTERNAL.md` | datasets, licences, papers, links, GitHub, artifacts |
| `08-PEOPLE.md` | team, roles, presentation, who has the GPU |
| `09-SESSION-LOG.md` | how we got here, and the mistakes worth not repeating |

## Working rules that produced the good results

1. **One variable per experiment.** Nearly every error below was caught because
   a second variable was absent.
2. **Keep a change only if the target metric improves without regressing STOI or
   SI-SAR, on BOTH eval sets.** A gain on one set is not a gain.
3. **Never rank on SI-SDR alone.** Split it into SI-SIR (noise removed) and
   SI-SAR (damage caused). That split produced most of our findings.
4. **Measure the ceiling before optimising toward it.** An oracle probe costs no
   training and kills bad plans in an hour instead of a week.
5. **Record negative results.** `docs/06-L1-artefact-evidence.md` is the
   project's most valuable document and it is mostly failures.
