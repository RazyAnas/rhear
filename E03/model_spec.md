# RHEAR E03 — L1 baseline model spec
**Status: specified and measured. NOT TRAINED.** Approve before any long run.

Every number below is measured by `python3 model/gtcrn_lite.py` and asserted in
`tests/test_model_causality.py`. Nothing is estimated except the Colab wall-clock,
which is labelled as an estimate.

---

## 1. Architecture — `GTCRNLite`

```
noisy waveform 16 kHz
        │
   STFT  n_fft 512 (32 ms), hop 64 (4 ms), asymmetric windows
        │  (real, imag, magnitude) → 3 channels
        ▼
   ERB band split   257 linear bins → 48 ERB bands     [fixed, 0 params]
        │
   encoder: 4 × causal depthwise-separable conv, channel shuffle
        │      3→16  (stride F 2)   48→24 bands
        │     16→24  (stride F 2)   24→12
        │     24→24  (stride F 2)   12→6
        │     24→32  (stride F 1)    6→6
        ├──────────────────────────► speech-presence head (1×1 conv → sigmoid)
        ▼
   dual-path RNN
        │  intra-frame GRU over FREQUENCY, bidirectional (no time lookahead)
        │  inter-frame GRU over TIME, unidirectional (causal)
        ▼
   decoder: 4 × causal transposed separable conv, skip-connected to the encoder
        │      time dimension trimmed from the FRONT to preserve causality
        ▼
   48 bands → 257 bins
        │
   bounded complex ratio mask
        │  |M| = tanh(·) ∈ [0,1)      ← cannot invent energy
        │  ∠M = atan2(·)              ← phase is predicted, not copied
        ▼
   Ŝ = |M|·|Y|·exp(j(θ_Y + ∠M))  →  iSTFT  →  enhanced waveform
```

Two deliberate choices worth defending:

* **The ERB split is fixed, not learned.** It costs no parameters and imposes the
  right frequency prior. Learning it would spend a third of the budget.
* **The frequency-axis GRU is bidirectional; the time-axis GRU is not.** Looking
  both ways across the spectrum of the *current* frame is free; looking forward
  in time is not, and `test_strictly_causal` asserts it does not happen.

## 2. Measured numbers

| | value | budget (`docs/02-architecture.md`) | |
|---|---|---|---|
| parameters | **22,988** (23.0 k) | ≤ 100 k | **PASS** |
| INT8 model size | **~23 KB** | ≤ 100 KB | **PASS** |
| MACs / frame | 236,712 | — | |
| MAC rate | **59.2 MMAC/s** | ≤ 60 | **PASS** |
| algorithmic latency | **8.0 ms** | ≤ 8 ms | **PASS** |
| input / output | 16 kHz mono | 16 kHz | — |
| frame / hop / synthesis | 512 / 64 / 64 (32 / 4 / 4 ms) | — | |
| frame rate | 250 fps | — | |
| causality | future→past delta **0.00e+00** | strictly causal | **PASS** |
| mask bound under 500× input | max |M| = 1.0000 | ≤ 1 | **PASS** |

For calibration: GTCRN is 48.2 k params / 33.0 MMAC/s at a 16 ms hop; LiSenNet is
37 k / 56 M MAC/s. This model is smaller in parameters and comparable in MAC rate
**because it runs at a 4× higher frame rate** to hit 8 ms latency.

### The one real trade-off, measured

| hop | bands | params | MMAC/s | latency | |
|---|---|---|---|---|---|
| 64 | 64 | 22,988 | 75.1 | 8.0 ms | over MAC budget |
| **64** | **48** | **22,988** | **59.2** | **8.0 ms** | **chosen** |
| 64 | 40 | 13,436 | 37.6 | 8.0 ms | headroom if needed |
| 128 | 64 | 22,988 | 37.5 | 12.0 ms | cheaper, misses the latency spec |

Latency is bought with MACs. 48 bands is the largest configuration that keeps
8 ms latency inside the 60 MMAC/s budget.

## 3. Loss

Per `docs/02-architecture.md` §7, with the intelligibility-first term:

```
L = λ₁·L_SI-SDR + λ₂·L_mag + λ₃·L_RI + λ₄·L_asym + λ₅·L_SPP

L_SI-SDR  = −10 log₁₀( ‖s_target‖² / ‖e_noise‖² )
L_mag     = E[ ( |S|^0.3 − |Ŝ|^0.3 )² ]
L_RI      = E[ | |S|^0.3 e^{jθ_S} − |Ŝ|^0.3 e^{jθ_Ŝ} |² ]
L_SPP     = BCE( q̂, 1[ |S|² > η|N|² ] )

L_asym    = β_over ·E[ max(0, +Δ)² ]      Δ = |S|^0.3 − |Ŝ|^0.3
          + β_under·E[ max(0, −Δ)² ]      β_over/β_under = ρ ∈ [3,10]
```

`ρ` is scheduled by the noise class, which the dataset manifest already records
per sample — so the schedule is available at training time without extra
labelling. λ's are learned via homoscedastic uncertainty rather than hand-tuned.

**No adversarial loss in v1.** It buys MOS on paper and its failure mode is
hallucinated speech, which is the worst possible failure in a defence
communication system.

## 4. Dataset it will train on

Produced by `RHEAR_E03_dataset.ipynb`. `small` profile: 1,500 train / 250 val /
250 test / 300 held-out 4-second pairs at 16 kHz — **2.6 h of mixtures** from
real speech, real environmental noise, real military noise and real measured RIRs.
Speaker-, noise-source- and room-disjoint splits, plus a held-out real-world test
set that is never used for training or tuning.

## 5. Estimated Colab runtime

Measured here: **30.4 clips/s** forward+backward on this laptop's CPU, batch 8,
4-second clips → **0.8 min/epoch** over 1,500 clips **on CPU alone**.

A T4 is faster, but I have not measured that ratio and will not assert one. The
honest conclusion is the useful one: **compute is not the bottleneck for a 23 k
parameter model.** Data loading and STFT will dominate, so the notebook should
pre-compute spectrograms once. Expect a 60-epoch run in **well under an hour**,
and treat that as an estimate until the first real run reports it.

## 6. What is deliberately not here

- No training loop is run.
- No quantisation yet (that is G4, and it needs a trained model).
- No claim about STOI/PESQ/SI-SDR. Those are G3's job and no number exists until
  the model is trained on the inspected dataset.
