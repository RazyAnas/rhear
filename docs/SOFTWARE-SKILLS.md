# RHEAR — software skills needed

What the software side of this project actually uses. Grouped by how soon you need
it. Nobody needs all of it; pick a column and go deep.

---

## The three software roles

| Role | Owns | Core skills |
|---|---|---|
| **DSP / audio** | L0 — the noise canceller | Python + NumPy/SciPy, adaptive filters, sample rates and delay |
| **ML** | L1 + L2 — the neural parts | PyTorch, spectrograms, masking, training discipline |
| **Embedded** | Getting it onto the chip | C, ESP-IDF, fixed-point, quantisation |

Everyone needs Level 1.

---

## Level 1 — everyone, first week

**Python.** The whole pipeline is Python. You need functions, classes, NumPy arrays,
file I/O, `argparse`. Not web frameworks, not async.

**NumPy.** Array slicing, broadcasting, dot products, `mean`/`sum` along axes. Most of
our DSP is 20 lines of NumPy. If `a @ b` and `x[1:] = x[:-1]` are unfamiliar, start here.

**Git.** Branch, commit, diff, read a log. We version everything including negative
results.

**Reading a measurement honestly.** The most valuable skill on this project and the
least taught. Concretely: knowing that an average hides its distribution, that a metric
going up is not proof the thing got better, and that "I expected X and measured
not-X" is a result worth keeping. We have killed four of our own ideas this way.

---

## Level 2 — DSP / audio track

**Digital signal processing basics**
- Sampling rate, why 48 kHz vs 16 kHz vs 192 kHz changes everything
- FIR vs IIR filters — one is a plain weighted sum of recent samples, the other feeds
  its own output back
- Convolution, and why it is just "slide and multiply"
- Group delay — how long a filter takes to pass a signal through
- The FFT / STFT — turning a sound into a picture of frequency over time

**Adaptive filtering** — the heart of L0
- LMS and NLMS: keep nudging filter weights to shrink the error
- **FxLMS / FxNLMS**: same, but corrected for the fact that the speaker and air
  change the sound on the way to the ear
- Why normalising by input loudness is itself a defence against gunshots

**SciPy.** `scipy.signal` — `lfilter`, `butter`, `coherence`, `welch`. That is most of it.

Read our `rhear/core/anc.py`. The whole canceller is one readable loop.

---

## Level 3 — ML track

**PyTorch**
- Tensors, shapes, `nn.Module`, forward pass, `loss.backward()`, optimisers
- Conv2d, GRU, LayerNorm — the layers we actually use
- Understanding a shape error. You will get many; they are the job

**Audio ML specifics**
- Spectrograms and complex numbers — magnitude and phase
- **Masking**: predict a number 0–1 per frequency band saying "keep this much".
  Because it is bounded, the model can only remove, never invent
- ERB / mel bands — grouping 257 frequency bins into 48, the way hearing works
- Loss functions, and why the loss you pick is the thing you actually get

**Evaluation discipline** — as important as modelling here
- STOI, PESQ, SI-SDR, and splitting SDR into SI-SIR (noise removed) and SI-SAR
  (damage caused)
- **Data leakage**: if a noise recording appears in both training and test, your
  numbers are fiction. We have a script that asserts this and it has caught real bugs
- Stratified reporting — by SNR bucket and by noise class, never one average

Read `E03/train_interim.py` and `E03/eval_stratified.py`.

---

## Level 4 — embedded track

**C.** Pointers, arrays, structs, fixed-size integer types. No C++ needed.

**ESP32-S3 / ESP-IDF**
- Building and flashing, `menuconfig`, FreeRTOS tasks, pinning work to a core
- **I2S** — the protocol that carries digital audio between chips
- **I2C / SPI** — for configuring the codec

**ESP-DSP and ESP-NN.** Espressif's hand-optimised maths libraries. Knowing that
`dsps_dotprod_s16` exists and runs ~4× faster than a plain C loop is the difference
between fitting on the chip and not.

**Fixed-point arithmetic.** int16 instead of float. Understanding Q15 format, overflow,
and why we care: the chip is roughly 40% faster in int16 than float32, measured.

**Model deployment**
- ONNX export, and reading the operator list that comes out
- INT8 quantisation, ideally quantisation-*aware* training
- Knowing which operators the target actually supports. Ours has 7 unsupported
  operations left; that list drives real architecture decisions

---

## What you can skip

No web development, no databases, no cloud, no Docker, no CUDA kernel writing, no
distributed training. It is one small model on one small chip.

---

## Honest order to learn in

1. Python + NumPy — a week
2. Pick a track
3. For ML: get PyTorch to train *anything* end to end before touching our code
4. For DSP: implement plain LMS on a toy signal before reading FxNLMS
5. For embedded: blink an LED on an ESP32-S3, then get I2S audio in and out, before
   touching the model

The single biggest predictor of being useful here is not knowing more libraries. It is
being willing to measure something, find out you were wrong, and say so.
