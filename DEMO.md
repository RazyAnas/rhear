# RHEAR — internal hackathon demo runbook
**5 minutes. One command. No internet required.**

```bash
./demo.sh          # then open http://127.0.0.1:8765
```

---

## What we can and cannot claim

Say this early — it is a strength, not a weakness, and it pre-empts the question
every technical judge will ask.

| | |
|---|---|
| **Working and measured** | the ANC control system, the causality analysis, the direction mechanism, the noise-scene engine, the hardware selection |
| **Built but not yet run** | the speech-enhancement dataset pipeline |
| **Specified but not trained** | the L1 speech-enhancement model — so **no STOI/PESQ number yet** |
| **Not bought** | any hardware |

If asked "so what's real?": *everything on the screen is computed by the running
pipeline. The simulator was validated against closed-form theory before we
trusted any number from it — it reproduces the ANC coherence bound to 0.06 dB.*

---

## The 5-minute script

### 1. The problem (30 s)
Armoured-vehicle and rotor noise destroys radio intelligibility. Classical LMS
ANC assumes stationary noise and fails on exactly the noise defence has.
**We can now show why, with measurements, not assertions.**

### 2. The architecture in one line (30 s)
Point at the signal-flow diagram. **The purple line is the whole idea:**
L2 writes *filter coefficients* at 62 Hz; no neural computation sits in the
microsecond loop. That is what makes a ₹15k headset feasible.

### 3. The live system (2 min) — the hook
Let it run. Things happen every few seconds:

| watch for | what to say |
|---|---|
| **scene changes** every 5 s (engine → rotor → wind → siren) | "the scene engine classifies and swaps the control filter live" |
| **attenuation jumps** to −18 to −24 dB on engine/rotor | "that's cancellation at the ear, measured, not simulated playback" |
| **PROTECT mode** (red) every ~13 s | "a 160 dB impulse — the controller freezes rather than letting a saturated mic rewrite the filter" |
| **causality margins flipping sign** as the head turns | "this is the part nobody else has" — see below |
| **active reference switching L→R** | "each ear picks whichever microphone is still causal" |

### 4. The three results that win it (90 s)

**a. The causality cliff — a hardware decision made for free.**
Wideband attenuation vs the electrical delay of the audio codec:

| part | delay | attenuation |
|---|---|---|
| ANC codec (ADAU1772) | 38 µs | **−13.7 dB** |
| generic 48 kHz codec | 619 µs | **−0.0 dB — the loop does nothing** |

*We derived the 619 µs ourselves and it matches the published 500–600 µs.
That is a ₹1,600 part decision settled in simulation before buying anything.*

**b. Neural filter selection beats classical adaptation where it matters.**
On non-stationary noise, in the 250 ms after a scene change:

| | attenuation |
|---|---|
| plain FxNLMS (the classical baseline) | −11.4 dB |
| **learned selection + FxNLMS** | **−20.7 dB** |
| oracle (perfect knowledge) | −23.4 dB |

**+9.3 dB, closing 77 % of the gap to an oracle, with a 1,764-parameter selector.**

**c. Direction is worth 2.4 dB — and not for the reason we expected.**
A headset has **four** microphone→ear paths, not two. At +90° the left cup's own
mic is 102 µs non-causal while the *right* mic leads by 496 µs. Using both takes
each ear from **38 % → 68 % of azimuth covered**.

### 5. Honest close (30 s)
*"Five of seven internal gates pass. The speech-enhancement model is specified
and measured — 23,000 parameters, 59 MMAC/s, 8 ms latency, strictly causal — but
not yet trained, so we are not quoting a STOI number we do not have. The dataset
pipeline that feeds it is built and licence-cleared."*

---

## Questions you will get

**"Is this simulated?"** Yes, and deliberately — we validated the simulator
against closed-form theory first (coherence bound to 0.06 dB; codec delay derived
independently). It has already found four bugs in our own design and made two of
our claims fail. That is what it is for.

**"Why no trained model?"** Because we refused to train on a dataset we had not
inspected. The pipeline is built, the licences are cleared, and the model is
measured. Training is roughly a week of the three we have left.

**"What's novel?"** Neural computation in the *coefficient* path, not the audio
path — so AI latency never violates acoustic causality. Plus IMU-fused causal
reference selection, which the room-ANC literature cannot do because it has no
head to instrument.

**"Will it run on a headset?"** The control loop runs at 40 % of its deadline
here. The model is 23 KB in INT8. An INT8 speech enhancer of this class has
already been published running on a chip inside a headphone.

**"What doesn't work?"** Wind. Broadband noise below the driver's roll-off is
close to uncancellable in this plant, and we say so on the dashboard rather than
hiding it.

---

## If something breaks

| symptom | fix |
|---|---|
| page blank | `./demo.sh` again; check `tail /tmp/rhear_twin.log` |
| "reconnecting" | the twin restarted (auto-reload). Wait 20 s, reload the page |
| numbers frozen | `pkill -f rhear_twin.py` then `./demo.sh` |
| everything fails | the experiments still run standalone: `python3 rhear/experiments/e06_filter_generation.py` (~20 s, prints the +9.3 dB result) |

**Backup:** `results/` holds the saved .npz for every experiment, and
`docs/05-digital-twin.md` has every number in this runbook with its derivation.
