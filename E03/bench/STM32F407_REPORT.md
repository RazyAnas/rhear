# STM32F407 deployment feasibility — frozen RHEAR E03 L1 + L2

## What this is, and what it is not

**This is a static analysis with a per-kernel cycle model. It is NOT an
on-target measurement.** STM32Cube.AI, the ARM toolchain, an emulator and an
F407 board are all absent from this machine (verified, not assumed):

```
stm32ai: no   arm-none-eabi-gcc: no   openocd: no   qemu-system-arm: no
X-CUBE-AI installed: no      STM32F407 board attached: no
```

So steps 1–2 of the brief ("export to an STM32-compatible INT8 model", "use
STM32Cube.AI/CMSIS-NN to generate the target implementation") are **partially
done**: the model is exported and quantised, the operator set and memory are
measured exactly, but no target code was generated and no cycle count was taken
on silicon.

**Measured exactly** (from the real exported graph): Flash, SRAM, activation RAM,
MAC counts, operator coverage.
**Modelled** (stated assumptions, bracketed pessimistic/optimistic): cycles and
latency. This is a per-kernel throughput model, not a parameter-count guess — but
it is still a model, and the GRU term is the least certain part of it.

---

## L1 — measured

| quantity | value |
|---|---|
| parameters | 22,988 |
| **Flash (INT8 weights)** | **22.4 KB** |
| Flash (fp32, for reference) | 89.8 KB |
| **peak activation RAM, 1 frame** | **5.3 KB** |
| MAC / frame | 236,712 |
| frame rate | 250 fps (4 ms hop) |
| **MMAC/s** | **59.2** |

Exported ONNX: 105 nodes, 24 distinct operators, single-frame streaming shape.

*(The INT8 ONNX file is 283.7 KB — larger than fp32's 123.4 KB — because dynamic
quantisation inserts Quantize/Dequantize nodes and ONNX protobuf overhead
dominates for a model this small. The deployable figure is the 22.4 KB of
weights, not the ONNX file size.)*

### Where the compute goes

| block | MAC/frame | share | kernel class |
|---|---|---|---|
| dprnn_freq | 57,600 | 24.3 % | **GRU** |
| STFT + iSTFT | 46,080 | 19.5 % | FFT |
| dprnn_time | 43,008 | 18.2 % | **GRU** |
| dec0–dec3 | 77,376 | 32.7 % | ConvTranspose |
| enc0–enc3 + spp | 12,648 | 5.3 % | Conv |

**GRU is 42.5 % of MACs. ConvTranspose is another 32.7 %. Neither has a CMSIS-NN
kernel.** Only ~5 % of the compute sits in the ops CMSIS-NN accelerates well.

---

## L1 — cycle model, per 4 ms frame @ 168 MHz

Includes the **complete** pipeline, not just the forward pass: windowing, forward
rFFT, magnitude, quantisation, the network, dequantisation, inverse rFFT,
overlap-add.

| | pessimistic | optimistic |
|---|---|---|
| network | 747,036 | 373,528 |
| STFT (pre) | 15,652 | 11,082 |
| iSTFT (post) | 18,734 | 13,138 |
| quant / requant | 35,507 | 11,836 |
| **total** | **816,929 cyc** | **409,584 cyc** |
| **time** | **4.86 ms** | **2.44 ms** |
| **CPU** | **121.6 %** | **61.0 %** |

Throughput assumptions (MAC/cycle): conv 0.55–1.10, conv1×1 0.70–1.30,
ConvTranspose 0.35–0.70, GRU 0.20–0.40. rFFT-512 taken at 8–10 k cycles
(CMSIS-DSP `arm_rfft_fast_f32` published range).

### The binding constraint is NOT the 8 ms latency

The 8 ms figure is *algorithmic* latency (synthesis window + hop). The real
deadline is the **4 ms frame period** — a new frame arrives every 4 ms, so the
whole pipeline must finish inside 4 ms or the system falls behind without bound.

**Pessimistic 4.86 ms > 4 ms period → cannot keep up.** It would still be "within
8 ms" on a single frame, which is exactly the trap of quoting latency instead of
throughput.

---

## L2 — measured MAC rates (from E05), modelled cycles

| workload | MMAC/s | CPU pess | CPU opt |
|---|---|---|---|
| GCC-PHAT bearing | 21.1 | | |
| coherence confidence | 7.7 | | |
| acoustic-state features | 3.0 | | |
| selector MLP (1,764 params) | 0.11 | | |
| **L2 total** | **31.9** | **38.0 %** | **19.0 %** |

L2 Flash: ~1.8 KB weights + FFT twiddles. SRAM: <8 KB.

---

## Combined system

| configuration | CPU pess | CPU opt | verdict |
|---|---|---|---|
| **L1 + L2 + codec-based L0** (our hardware design) | **159.7 %** | **80.0 %** | **MARGINAL** |
| L1 + L2 + MCU-based L0 (no ANC codec) | 243.1 % | 128.7 % | **INFEASIBLE** |

48 kHz I2S acquisition itself is DMA (~0 % CPU), but the 48→16 kHz decimation for
L1 adds a further ~2–4 % that is not in the table.

---

## PASS / FAIL

| requirement | limit | measured / modelled | verdict |
|---|---|---|---|
| L1 Flash | 1 MB | **22.4 KB** | **PASS** (45× margin) |
| L1 activation RAM | 192 KB SRAM | **5.3 KB** | **PASS** |
| L1 + L2 + buffers SRAM | 192 KB | ~40 KB | **PASS** |
| L1 within 8 ms latency | 8 ms | 2.44–4.86 ms | **PASS** |
| **L1 within the 4 ms frame period** | 4 ms | **2.44–4.86 ms** | **MARGINAL / FAIL** |
| **L1 + L2 CPU headroom** | <100 % | **80–160 %** | **MARGINAL / FAIL** |
| **operator support** | CMSIS-NN | **12 of 24 ops unsupported** | **FAIL** |
| L0 coexistence, codec-based | — | +0.1 % | **PASS** |
| L0 coexistence, MCU-based | — | +49–84 % | **FAIL** |

---

## Cause of failure

Not one cause — but they rank clearly:

**1. Unsupported operators (hard blocker, not a tuning issue).**
12 of 24 operators have no CMSIS-NN kernel, covering 25 of 105 nodes and ~75 % of
the compute:

- **GRU ×2** — CMSIS-NN 4.x has LSTM but *no GRU kernel*. 42.5 % of MACs.
- **ConvTranspose ×4** — no kernel at all. 32.7 % of MACs.
- **Einsum ×2** — the ERB split/inverse; must be rewritten as a sparse matmul.
- **LayerNormalization ×2, Pow ×5, Sqrt ×2, Div, Atan** — transcendentals.
- **IsNaN / Where / Greater / Less ×6** — control flow that should not be in an
  inference graph at all; these come from `atan2` and clamp guards and are an
  export artefact worth removing regardless of target.

X-CUBE-AI covers more than raw CMSIS-NN (it does support GRU), but it generates
generic code for those layers — which is what the pessimistic GRU throughput of
0.2 MAC/cycle already assumes.

**2. Compute.** 59.2 MMAC/s for L1 plus 31.9 for L2 is 91 MMAC/s of mostly
SIMD-hostile work on a 168 MHz M4. Even optimistically that is 80 % CPU with no
margin for jitter, interrupts or the decimator.

**3. Preprocessing is NOT the problem** — 8 % of the frame budget. Worth stating
because it is the usual suspect.

**Not RAM. Not Flash.** Both pass with large margin (45× on Flash, 36× on SRAM).
Anyone sizing this from parameter count alone would have concluded "22 KB, fits
easily" and been wrong for reasons that have nothing to do with size.

---

## Recommendation on the processor

**Do not commit to STM32F407 yet, and do not abandon it on this analysis alone.**

The honest position is that F407 is *marginal in the optimistic case and
infeasible in the pessimistic one*, and the spread between those two is driven by
one number — GRU throughput — which is precisely the number this analysis cannot
measure without a board.

Two things should happen before the processor is chosen:

1. **Measure the GRU kernel on target.** One F407 board and a benchmark of the
   two GRU layers would collapse the 80–160 % range to a single figure. That is
   the whole decision.
2. **Consider removing the GRU before changing the processor.** It is 42.5 % of
   MACs, it is the unsupported op, and it is the least certain cost. A depthwise
   temporal convolution in its place would be CMSIS-NN-native and would very
   likely bring the pessimistic case under 100 %. That is a model change, so it
   belongs in the E03 experiment queue, not here — but it may be cheaper than a
   more expensive part.

If both fail, the STM32F4 → **STM32H7** step (480 MHz Cortex-M7, double-issue,
larger caches) gives roughly 4–6× the throughput and would move this comfortably
into feasible. STM32N6 with its NPU is the other option and was already the
target-build choice in `docs/04-hardware-design.md`; F407 was only ever the
prototype board.
