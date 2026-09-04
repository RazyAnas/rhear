# Demo assets for the 7th

Plan: `~/.claude/plans/im-not-getting-3-glistening-scroll.md`.
Full hardware walkthrough: `docs/PHASE0-STEP-BY-STEP.md`.

**Use `/opt/anaconda3/bin/python`.** It is the only interpreter on this machine
with numpy, scipy, torch, soundfile and pyserial. `python3` on the PATH has none
of them.

---

## Item 1 — the on-cup coherence measurement (do this first)

This is the answer to "you don't have the codec": we measure the physical
ceiling of the actual headset, which is the part of L0 that is novel anyway.

### Wire it (`docs/PHASE0-STEP-BY-STEP.md` §5.1b, §5.2)

| MAX4466 | ESP32-S3 | where |
|---|---|---|
| VCC / GND | 3V3 / GND | |
| mic 1 OUT | **GPIO 1** | **reference** — taped to the OUTSIDE of the X3A shell |
| mic 2 OUT | **GPIO 2** | **error** — inside the cup, wire threaded under the cushion |

**Do not drill the muffs.** Aim for ~7 cm between the two mics through the
shell — that spacing is where the 204 µs of acoustic travel, and therefore the
146 µs causality budget, comes from. Set both trimmers to roughly the same gain;
the script prints the ratio and tells you if it is off.

### Flash

`esp32_dual_adc/esp32_dual_adc.ino` — Arduino IDE, board "ESP32S3 Dev Module".
It streams tagged 12-bit pairs at a hardware-paced 8 kHz. (The snippet in the
docs used `delayMicroseconds`, so its real sample rate was unknown and drifting;
a wrong fs makes coherence read *low*, which would have looked like a physics
result and been a timing bug.)

### Record and analyse

With broadband noise playing about a metre away:

```bash
/opt/anaconda3/bin/python demo/coherence.py capture --seconds 20 --out demo/captures/cup
```

Re-run the numbers later, no hardware needed — this is what you run on stage:

```bash
/opt/anaconda3/bin/python demo/coherence.py analyse demo/captures/cup.npz
```

Outputs `cup_coherence.png` (the slide), `cup_refA_errB.wav`, and a per-band
table of γ² and the ceiling `NR ≤ −10·log₁₀(1−γ²)`.

The script refuses to quietly give you a bad number: it checks for clipping,
for a nearly-silent channel, for mismatched mic gains, for dropped USB bytes,
and for a firmware/host sample-rate mismatch. Fix what it flags before trusting
the result.

**Report the cup number, not a free-air number.** Free air always looks better
and it is not the system. A low number is a finding, not a failure — say it.

`demo/captures/_synthetic.npz` is a fake capture for rehearsing the `analyse`
step without hardware. It is **not** a measurement; never show it as one.

---

## Item 3 — on-chip proof

```bash
cd psram_test && idf.py -p <PORT> flash monitor
```

Show the serial output: per-layer error vs PyTorch under `2e-5`, per-stage µs,
`RESULT: PASS`. Say exactly what it is — STFT → ERB → 4 encoder blocks →
fullband → fuse → SPP, the front half of G7, bit-accurate on the target silicon.
The DPRNN and decoder are not ported yet; `E03/g7_esp32_preflight.json` is the
measured budget that says they fit (32.34 MMAC/s against ~200 per core).

Do not overclaim this as full on-chip inference.

---

## Item 4 — L2 digital twin

```bash
/opt/anaconda3/bin/python rhear_twin.py     # localhost:8765, ~15 s to warm up
```

Start it before you walk in.

---

## Item 2 — L1 live (build on the 5th)

INMP441 → ESP32-S3 → G7 → MAX98357A → 16 Ω speaker in the cup.
Fallback pre-rendered A/B pairs already exist in `E03/runs/demo_ab/`.
Demo with **G7-base**, `E03/runs/g7_hop256_50k/best.pt`.
