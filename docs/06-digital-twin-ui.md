# RHEAR — Digital Twin UI
### The live visualisation layer, from simulation to headset
v0.1 · 2026-08-27 · `python3 rhear_twin.py` → http://127.0.0.1:8765

---

## 1. What this is

A **persistent, live** view of the running RHEAR pipeline — not a results dashboard. It is
the visualisation layer for the whole programme: the same UI will show the simulation today,
the laptop rig next, the dev board after that, and the final headset last. Nothing in the
frontend changes across that transition.

**Every displayed number is computed by the pipeline.** Where a subsystem is not implemented
(L1 speech enhancement, the boom mic) its block reports `idle` and its scalars are simply
absent, so the UI shows `--`. Nothing is synthesised to populate a panel.

It is served locally rather than published as a hosted page, because a hosted artifact's
content-security policy blocks connections back to a local pipeline — it could not be
genuinely live.

## 2. The contract — `rhear/telemetry/schema.py`

One dataclass, `TelemetryFrame`, is the entire interface between producers and the UI.

```
TelemetryFrame
  t, seq, source, experiment, mode, schema
  channels : {name -> Channel}    time-domain taps      (rate, samples, clipped)
  spectra  : {name -> Spectrum}   frequency views       (f0, df, mag_db)
  scalars  : {name -> Scalar}     any single number     (value, unit, lo, hi, good, group)
  vectors  : {name -> Vector}     any array to plot     (values, kind: line|bar|stem)
  blocks   : {node -> Block}      flow-diagram state    (state, load, latency_us, rate_hz)
  edges    : [Edge]               flow activity         (src, dst, level)
  geometry : Geometry             head/source view
  events   : [Event]
```

**Why this shape.** The frontend renders `scalars`, `channels`, `spectra` and `vectors`
*generically*: a `Scalar` carries its own unit, expected range, grouping and whether it is
currently good, so the UI can draw a tile with a bar and a colour without knowing what the
quantity means. **A new experiment that publishes a new scalar gets a new tile with no
frontend change.** That is the requirement about future experiments appearing automatically,
discharged by the data model rather than by a plugin system.

Adding a quantity: put it in `scalars`/`vectors`/`channels`. Adding a *kind* of quantity:
bump `SCHEMA_VERSION`. The version is displayed in the header, so a stale UI is visible.

## 3. Producers — `rhear/telemetry/sources/`

| source | status | what it is |
|---|---|---|
| `simulation.SimulationSource` | **running** | the real RHEAR pipeline in real time |
| `hardware.HardwareSource` | **stub, deliberate** | newline-JSON over USB CDC. Publishes *nothing* until a device is attached, so the dashboard reads "no device" instead of stale or invented data |

All subclass `TelemetrySource`. When the MCU starts emitting frames, **only `hardware.py`
changes** — schema, bus, server and UI are already correct.

### What the simulation source actually runs

Not a mock. Per 21.3 ms block, at 48 kHz:

| stage | code | rate |
|---|---|---|
| scene generation, head rotation | `core/streaming.py` | 48 kHz |
| plane-wave rendering, fractional time-varying delay | `sim/geometry.py` | 48 kHz |
| microphone AOP saturation, self-noise | `sim/electronics.py` | 48 kHz |
| **L0 FxNLMS + secondary path** | `core/l0.py` (shares `score()` with `core/anc.py`) | 48 kHz |
| impulse detector | `core/anc.py` | 48 kHz |
| **L2** acoustic state, GCC-PHAT bearing, gyro fusion, coefficient selection | `core/state.py`, `core/doa.py` | 46.9 Hz |
| causality margin per cup → engage/disengage | `sim/geometry.py` | 46.9 Hz |

Measured on this laptop: **real-time factor ≈ 0.31–0.45**, L0 load ≈ 28–43 % of its block
deadline. It is genuinely live, with headroom.

## 4. Panels

Signal flow (blocks light by state, **the coefficient path drawn distinctly from audio** —
that distinction is the architecture); per-microphone waveforms with saturation flags;
scrolling residual spectrogram; reference-vs-residual spectrum; head-and-source view with the
measured causal cones, true vs estimated bearing and the active reference; grouped scalar
tiles; vector plots (filter coefficients, acoustic state, filter-bank scores); event log.
Header carries the system mode, source kind, connection state, schema version, pause and a
time scrubber over the last 900 frames. **Default is live.**

System modes: `NORMAL / ENGINE / ROTOR / WIND / IMPULSE / PROTECT / LOW_CONFIDENCE`.

## 5. Four bugs the twin exposed on first run

Building it against the real pipeline immediately surfaced things the offline experiments had
not, which is the argument for having it.

1. **numpy scalars broke the wire format.** `/api/meta` worked while `/api/history` and the
   SSE stream silently returned nothing. Fixed by coercing at publish time, so no producer —
   including a future MCU — can break the format by leaking a type.
2. **The filter bank was trained against a different plant than the live loop ran.** Bank
   training used `primary_path`; the live loop rendered through the head geometry and cup
   filter. Engine and rotor still worked (a periodic signal absorbs a delay mismatch —
   E01/F3 again) but **wind and siren came out at +6 dB: the controller was adding noise.**
   Both now share one `_render_static()`.
3. **L=512 was too short**, so the 50 Hz engine filter trained to +1.06 dB. E01/F4 predicted
   it: one period is 960 samples. The rule has now caught the same mistake in two experiments
   and the twin.
4. **Wind at 60–900 Hz cannot be cancelled in this plant at any step size** (best −1.07 dB,
   worst +4.24 dB). Not actuator bandwidth — engine has 85 % of its energy below the driver's
   150 Hz corner and still reaches −24 dB. The mechanism is that below the roll-off `1/S` is
   ill-conditioned, and a *broadband* source needs that inverse across a continuum where a
   *periodic* one needs it at a few discrete frequencies. The wind scene now sits inside the
   actuator band, and per-scene step sizes come from a measured sweep — which is L2 head C
   doing its designed job.

Live per-scene attenuation after those fixes: **engine −18.8 dB, rotor −24.2 dB, siren
−2.7 dB, wind −0.7 dB.** Wind and siren are weakly controllable here and the dashboard says
so rather than hiding it.

## 6. Plugging in the next thing

| next | what changes |
|---|---|
| **E03** speech enhancement | publish `channels["boom"]`, `channels["enhanced"]`, `scalars["stoi"|"pesq"|"si_sdr"]`, set `blocks["l1"]` active. No frontend change |
| **G5** direction | already published (`geometry`, bearing scalars). Fixing the two blockers changes only the source |
| **laptop rig** | new `sources/rig.py` reading the audio interface; same frame |
| **dev board / headset** | `sources/hardware.py`, `--port /dev/tty.usbmodem*`; same frame |

## 7. Running

```bash
python3 rhear_twin.py                    # simulation (default)
python3 rhear_twin.py --source hw --port /dev/tty.usbmodemXXXX
```

`--publish-hz` sets the display rate only; the DSP always runs at full rate. First launch
trains the filter bank (~40 s) and caches it to `results/twin_bank.npz`.
