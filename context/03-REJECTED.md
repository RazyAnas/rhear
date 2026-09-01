# What we tried and killed

Full evidence lives in `docs/06-L1-artefact-evidence.md`. This is the index.
**Do not re-run these without a new hypothesis that predicts a different result.**

| # | idea | verdict | why |
|---|---|---|---|
| H9 | comb / harmonic post-filter | REJECTED | +0.004 PESQ. Nothing. |
| H10 | inference-time mask post-processing | REJECTED | **all 13 settings lowered SI-SAR** — floors, power compression, temporal smoothing |
| G3 | structured pruning | REJECTED | 5% masking costs 2.08 dB SI-SAR; and at 25k params we are already half GTCRN's size. Bottleneck is operator support, not parameter count |
| — | learned phase branch | REMOVED, confirmed | +0.023 STOI / +0.075 PESQ / +1.34 dB SI-SAR when removed. `atan2` applied one angle across bands up to 19 bins wide, while phase wraps faster. Also drops `Atan`, which CMSIS-NN cannot run |
| G1 | perceptual loss, weight 1.0 | REJECTED | worse on **every metric on both sets**. Probe showed the term sits near 14 while base loss goes negative — at w=1.0 it is ~7x the objective and replaces it |
| G5 | perceptual loss, weight 0.15 | REJECTED (weak) | everything flat; PESQ down slightly. Weak test — `best.pt` was epoch 0 |
| H11 | triangular ERB synthesis | REJECTED | PESQ up (+0.028/+0.061) as the papers predict, but SI-SAR down 1.63/1.20 dB, and SAR is our binding constraint |
| — | direction as a separate mechanism | REJECTED | coherence dominates: +0.03 gamma^2 buys +3.3 dB, 4 ms of lead buys 0.3 dB |
| — | accelerometer lead time | REJECTED | same reason as above |
| — | network inside L0's audio path (Deep ANC) | REJECTED by design | cannot meet the 146 us causality budget. This is a *feature* of our architecture, not a limitation — cite it |

## Two that are open, not rejected

**G6 — deep-filtering cascade.** Built, works, fits (26,479 params, 97.3 MMAC/s,
core 1 at 119.3/200, **no new ONNX operators**). 9 of 10 metric cells improved;
the one failure was SI-SAR −0.03 dB on E_def, which is noise. But it captured
only **~3% of the +0.566 PESQ headroom** the ceiling probe said it opened, on 5
epochs of fine-tuning. Val SI-SDR was **still rising at the last epoch**.
**Undertrained, not refuted.** Needs a full-length run on a real GPU.

**Perceptual loss generally.** Bracketed but not eliminated: harmful at 1.0,
inert at 0.15. MetricGAN+ was deliberately deferred by the user — see if
capacity/frame-rate changes pay first.

## The pattern in these failures

H11, G1, G5 and arguably G6 were all **efficiency plays against a ceiling too
low to matter**, or ceiling plays while sitting at 60% of the ceiling. The
cross-benchmark (`02-RESULTS.md`) later showed a model twice our size scores the
same as us on our data — so much of the gap we were attacking was never ours.

**Check your target against an external reference before optimising toward it.**
That one measurement would have saved several days.
