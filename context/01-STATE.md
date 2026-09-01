# Where everything stands

## Current best model

**G7** — `E03/runs/g7_hop256_50k/best.pt`, **49,663 params, hop 256**.
Architecture: no phase branch, full-band + sub-band, bounded tanh mask.
Flags: `--hop 256 --ch 32,48,48,64 --no-phase --fullband`

| | STOI | PESQ | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|
| E_def (300 clips) | 0.8151 | 1.6351 | 10.05 | 16.47 | 11.91 |
| realnoise (300) | 0.8428 | 1.6128 | 9.61 | 16.01 | 11.65 |
| **PS target** | **0.85** | **2.5** | **15** | — | — |

G7 **passed the decision rule on both sets** — every metric up vs G2, the first
clean KEEP after H11, G1, G5 and G6 all failed. The quality gain is small
(+0.006 / +0.020 PESQ); **the real result is the cost**:

| | G2 (previous best) | **G7** |
|---|---|---|
| params | 24,975 | **49,663** |
| compute | 62.74 MMAC/s | **32.34 MMAC/s** |
| core 1 | 84.8 / 200 | **54.3 / 200** |
| training | ~25 min/epoch | **4.7 min/epoch** |

Same quality, **half the inference compute**, 5x faster experiments. We had been
running the network 4x more often than any comparable published system purely
because our hop was 64. Cost: L1 latency 4 ms -> 16 ms frames, on the comms path
only — **L0's 146 us budget is untouched**.

Previous checkpoints kept for comparison: `g2_fullband` (24,975, hop 64),
`g6_deepfilter2` (26,479, deep-filter stage, undertrained).

## Against the targets, honestly

- **STOI 0.843 vs 0.85** — close, and we **already exceed 0.85 at every input
  SNR at or above 5 dB** (0.900 / 0.919 / 0.958 on G2; G7 is marginally better).
  The average is dragged under by the third of the test set deliberately placed
  below 0 dB.
- **PESQ 1.64 vs 2.5** — the real shortfall, but see the cross-benchmark: a
  published model twice our size scores 1.643 on the same clips.
- **SNR** — quote **both** and say which is which. SI-SIR (noise removed) is
  16.5 dB and clears 15. SI-SDR (net, after our own distortion) is 10.05 and
  does not. Picking the flattering one will not survive questions.

## What G7 unlocks — do this next

Core 1 sits at **54.3 of 200**. G7's validation curve was **still rising at
epoch 23**, so there is headroom. In order:

1. **~75-100k params at hop 256.** The user's rule was 50k first, higher only if
   50k still shows headroom. It does.
   `--hop 256 --ch 40,60,60,80 --no-phase --fullband` is ~64.7k at 43 MMAC/s.
2. **Re-run the deep-filter stage at hop 256.** G6 cost +34.6 MMAC/s and was
   undertrained at 5 epochs; at 4.7 min/epoch a full 24-epoch run now costs
   under 2 hours and fits the budget comfortably.
3. **Track B — larger corpus.** The user authorised non-commercial datasets
   ("its just project not for commercial use just a demo"), unlocking ESC-50,
   UrbanSound8K (CC BY-NC) and DEMAND (CC BY-SA). Changes the dataset, so it
   invalidates frozen baselines — its own step.
4. **Rebuild the demo from G7**: `python3 E03/build_demo_ab.py --ckpt runs/g7_hop256_50k/best.pt`
   then push (see `07-EXTERNAL.md` — the GitHub copy is behind).
5. **Hardware (G4).** Purchase decision only. 41-week codec lead time.

## The strategic picture

Two independent constraints, and only one of them is ours:

| | |
|---|---|
| our PESQ ceiling (perfect mask, 48 ERB bands, noisy phase) | **2.731** |
| ceiling with a deep-filter cascade | **3.297** |
| fraction of its ceiling G2 achieves | **60%** |
| fraction needed for PESQ 2.5 | 92% today, 76% with the cascade |

**Raising the ceiling alone does not get there** — G6 moved the ceiling 21% and
the result 1%. The binding problem is how little of the representation the
training extracts, and the cross-benchmark says a state-of-the-art model of
twice our size extracts no more of it on our data.

## What to do next, in order

1. **Evaluate G7** (above). Decide keep/reject on PESQ + STOI + SI-SAR.
2. **If G7 holds at half the compute, adopt it as the new baseline.** Then
   optionally test ~75-100k params, which the freed budget now allows.
3. **Track B — larger/more diverse training corpus.** The user has explicitly
   authorised non-commercial datasets ("its just project not for commercial use
   just a demo"), which unlocks **ESC-50** and **UrbanSound8K** (CC BY-NC) and
   **DEMAND** (CC BY-SA), previously excluded on licence grounds. This changes
   the dataset, so it invalidates frozen baselines — do it as its own step.
4. **Full-length deep-filter run.** G6 got 5 epochs of fine-tuning; DeepFilterNet
   trains this stage far longer. Needs the friend's GPU, not the laptop.
5. **Hardware (G4).** Nothing bought. The ADAU1772's manufacturer lead time is
   **41 weeks**, so if stock clears, the next batch lands after the deadline.
   This is a purchase decision only — the design is verified.

## Deliberately NOT doing

- More mask post-processing sweeps (H10 killed 13 settings)
- More perceptual-loss weight hunting (G1 at 1.0 harmful, G5 at 0.15 inert)
- Pruning (G3: our model is already half GTCRN's size)
- Putting a network inside L0's audio path (cannot meet 146 us)
