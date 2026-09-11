# Putting G13b on the ESP32 — staged port

The model is **not** on the chip yet. This directory is the work to get it
there, staged so each piece is numerically proven against PyTorch before the
next one is built on it. That discipline is why `psram_test.c`'s existing half
can be trusted to 2e-5, and it is the only way a hand-written port ends up
correct rather than plausible.

## Where it stands

| stage | what | status |
|---|---|---|
| 0 | weight export to a flat f32 blob + offset table | **DONE** — `export_weights.py`, 99,730 floats, 389.6 KB fp32 / **97.4 KB int8** |
| 1 | **bidirectional GRU** (the hardest piece) | **DONE and VALIDATED** — `gru.c`, −133.5 dB relative error, max abs 9.0e−07 vs PyTorch |
| 2 | LayerNorm + the two Linears → complete DPRNN | not started |
| 3 | encoder at 96 bands (psram_test.c has it at 48) | not started |
| 4 | decoder: 4 ConvTranspose blocks + U-Net skips | not started |
| 5 | ERB inverse, mask apply | not started |
| 6 | STFT / ISTFT on-chip (psram_test.c reads a precomputed spectrum) | not started |
| 7 | wire into the I2S loop, measure µs/frame on silicon | not started |

## Why the GRU first

`dprnn` is 27,872 of the model's parameters — the largest single block and the
one `psram_test.c` stopped short of. It is also the easiest to get subtly
wrong: PyTorch's reset gate multiplies the **hidden** projection only,

    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))

and several frameworks instead apply `r` to `h` before the matmul. That
variant produces output that looks entirely reasonable and is numerically
different. Validating it alone, against a PyTorch reference on random input,
is what separates "compiles" from "correct".

`gru_f` is also **bidirectional** — over the frequency axis, not time, so
causality is preserved, but it is two passes and twice the weights.

## Feasibility, from the numbers we have

- **Memory:** 97.4 KB at int8 against the PS's 200 KB cap; 389.6 KB as fp32,
  which fits PSRAM comfortably. Not a constraint either way.
- **Compute:** ~26–32 MMAC/s against roughly 200 MMAC/s available on one
  ESP32-S3 core with esp-dsp. Roughly 15% of a core. Not a constraint.
- **The constraint is implementation effort**, not silicon.

## Reproduce

    /opt/anaconda3/bin/python export_weights.py     # weights + reference tensors
    clang -O2 -std=c11 -o /tmp/gru gru.c -lm && /tmp/gru
