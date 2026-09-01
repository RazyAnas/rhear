# Every number

All figures: 300 held-out clips per set, sources verified disjoint from
training. E_def = 8 defence noise classes. realnoise = environmental.

## Model progression

### edef

| run | params | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR | note |
|---|---|---|---|---|---|---|---|
| (noisy input) | — | 0.7688 | 1.2915 | 2.50 | — | — | what we start from |
| H3 | 22,988 | 0.7876 | 1.5289 | 9.59 | 20.18 | 10.50 | phase branch present, 20k data |
| G0 | 22,956 | 0.8102 | 1.6043 | 9.92 | 16.28 | 11.83 | phase branch REMOVED |
| G1 | 22,956 | 0.7916 | 1.4927 | 8.89 | 14.87 | 11.12 | + perceptual loss w=1.0 |
| G5 | 24,975 | 0.8134 | 1.6167 | 10.07 | 16.60 | 11.82 | + perceptual loss w=0.15 (fine-tune) |
| G2 | 24,975 | 0.8141 | 1.6295 | 10.05 | 16.40 | 11.89 | + full-band branch  <-- BASELINE |
| G6 | 26,479 | 0.8155 | 1.6416 | 10.15 | 16.77 | 11.86 | + deep-filter cascade |
| **PS target** | | **0.85** | **2.5** | **15** | | | |

### realnoise

| run | params | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR | note |
|---|---|---|---|---|---|---|---|
| (noisy input) | — | 0.7989 | 1.2951 | 1.80 | — | — | what we start from |
| H3 | 22,988 | 0.8135 | 1.5095 | 8.95 | 19.97 | 10.15 | phase branch present, 20k data |
| G0 | 22,956 | 0.8360 | 1.5683 | 9.17 | 15.10 | 11.55 | phase branch REMOVED |
| G1 | 22,956 | 0.8143 | 1.4375 | 7.62 | 12.64 | 10.87 | + perceptual loss w=1.0 |
| G5 | 24,975 | 0.8389 | 1.5875 | 9.40 | 15.56 | 11.61 | + perceptual loss w=0.15 (fine-tune) |
| G2 | 24,975 | 0.8393 | 1.5927 | 9.35 | 15.37 | 11.61 | + full-band branch  <-- BASELINE |
| G6 | 26,479 | 0.8409 | 1.6141 | 9.50 | 15.64 | 11.67 | + deep-filter cascade |
| **PS target** | | **0.85** | **2.5** | **15** | | | |
## G7 — the current best (hop 256, 49,663 params)

| set | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def | 0.8151 | 1.6351 | 10.05 | 16.47 | 11.91 |
| realnoise | 0.8428 | 1.6128 | 9.61 | 16.01 | 11.65 |

vs G2: every metric up on both sets. **PASS** on the decision rule.
Cost: **32.34 MMAC/s vs 62.74**, core 1 **54.3/200 vs 84.8**, **4.7 min/epoch
vs ~25**. Same quality, half the compute, 5x faster experiments.

## THE CROSS-BENCHMARK — the most important measurement in the project

GTCRN (Rong et al., ICASSP 2024) is peer-reviewed, **48,245 params (1.93x ours)**,
and published at **PESQ 2.87 / STOI 0.940 on VoiceBank+DEMAND**. We ran its own
released weights through its own published inference path (16 kHz, n_fft 512,
hop 256, sqrt-Hann) over **our** 300 defence clips, scored by **our** scorer.

| model | params | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| (noisy input) | — | 0.7688 | 1.2915 | 2.50 | — | — |
| GTCRN trained on VCTK-DEMAND | 48,245 | 0.7648 | 1.4442 | 6.45 | 12.19 | 9.70 |
| GTCRN trained on DNS3 | 48,245 | 0.8182 | 1.6430 | 9.61 | 18.32 | 10.79 |
| **RHEAR G2** | **24,975** | 0.8141 | 1.6295 | 10.05 | 16.40 | **11.89** |
| **RHEAR G6** | **26,479** | 0.8155 | **1.6416** | **10.15** | 16.77 | 11.86 |

**The same weights that score 2.87 on VoiceBank+DEMAND score 1.643 on our data.**
A 1.23 PESQ loss from changing nothing but the test set.

Against their stronger checkpoint we are level (dPESQ −0.001, dSTOI −0.003) and
**ahead on SI-SDR +0.53 and SI-SAR +1.07 with half the parameters**. Their
VCTK checkpoint actually *lowers* STOI on defence noise (0.7688 -> 0.7648).

Why our set is harder: VoiceBank+DEMAND tests +2.5 to +17.5 dB SNR against
domestic noise and starts at noisy PESQ ~1.97. Ours spans −10 to +20 dB with a
**third of clips below 0 dB**, against gunshots, artillery and rotors, starting
at **1.29**.

**Reproduce:** `python3 E03/crossbench.py --ckpt dns3` (~3 min, no training).
Weights auto-cloned from https://github.com/Xiaobin-Rong/gtcrn

## Stratified by input SNR (G2, E_def) — the strongest slide we have

| bucket | n | STOI | PESQ | vs 0.85 |
|---|---|---|---|---|
| < −5 dB | 49 | 0.618 | 1.170 | fails |
| −5..0 dB | 52 | 0.699 | 1.192 | fails |
| 0..5 dB | 44 | 0.795 | 1.409 | fails |
| 5..10 dB | 66 | **0.900** | 1.683 | **PASSES** |
| 10..15 dB | 53 | **0.919** | 2.010 | **PASSES** |
| > 15 dB | 36 | **0.958** | 2.490 | **PASSES** |
| ALL | 300 | 0.814 | 1.629 | |

We meet the intelligibility target at **every SNR at or above 5 dB**. PESQ never
passes, topping out at 2.49.

## Oracle ceilings — what any model of each family could reach

`E03/oracle_ladder.py`, 40 clips, deep filter order N=5. An oracle cannot be
beaten by any model of its family, so a family whose oracle sits near 2.5 is
finished before it starts.

| family | STOI | PESQ | SI-SDR | SI-SAR |
|---|---|---|---|---|
| A noisy | 0.761 | 1.250 | 2.14 | — |
| **E — G2 as trained** | 0.795 | **1.527** | 8.81 | 10.92 |
| B ideal mask @48 bands, noisy phase | 0.956 | **2.731** | 13.18 | 14.71 |
| F ideal mask @257 bins, noisy phase | 0.966 | 3.179 | 14.33 | 15.66 |
| **H ERB + deep-filter cascade (96 ms)** | 0.968 | **3.297** | 17.37 | 17.81 |
| C ideal @48, clean phase | 0.987 | 4.070 | 22.20 | 22.44 |
| D ideal @257, clean phase | 0.996 | 4.511 | 25.72 | 25.97 |

**Required efficiency against PESQ 2.5:** 92% of oracle today, 76% with the
cascade. **We achieve 60%.** Raising the ceiling is necessary and not sufficient.

*Caveat:* at a 32 ms fit window the oracle is near-degenerate (8 equations, 5
unknowns) and reports an implausible 4.276 — discard it. The 96 ms and 192 ms
numbers are defensible; 192 ms (3.073) is the conservative one to quote.

## Timing budget — ESP32-S3, both cores

| component | MMAC/s | core |
|---|---|---|
| L0 FxNLMS, L=128, stereo @192 kHz | 172.0 | 0 |
| audio I/O (DMA/ISR) | 6.4 | 0 |
| L1 speech enhancer | 62.7 (G2) | 1 |
| L2 features + ACF | 13.8 | 1 |
| L2 predictability solve | 8.2 | 1 |
| **total** | **263 of 400.3 — 66%** | **PASS** |

The L2 solve is 8.2 only because `prediction_floor_db` uses **Levinson-Durbin**.
A generic O(n^3) solver on the 256x256 Toeplitz system costs **350 MMAC/s alone**
and the budget FAILS at 150%. 43x ratio. That one function decides whether the
headset fits.

## Other measured results worth keeping

- **E06 filter selection** (the L2 claim): learned selection + FxNLMS refine
  gives **−12.19 dB overall and −20.67 dB in the 250 ms after a noise change**,
  vs −8.11 / −11.39 for plain FxNLMS. **+9.3 dB at transitions**, closing 77% of
  the gap to an oracle. Selector is **1,764 params, 83.2% frame accuracy**.
- **E08 filter length**: shortening L0 to L=128 at 192 kHz costs **7.5 dB on a
  200 Hz tonal rotor component and ~0 dB on engine, hull and wideband**. Do not
  hide the 7.5 dB. Say "L2 retuning may partially recover this", never "solves".
- **SI-SAR ceiling probe**: 22.91 dB achievable on our own 48-band grid vs 10.5
  achieved — this is what proved the artefact problem is training, not
  representation.
