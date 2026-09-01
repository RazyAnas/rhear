# RHEAR — every source we used

SIH 2026 · PS 26052 · DRDO / iDEX

For each: **what it says**, **why we used it**, **how it helped solve the
problem**. Grouped by the decision each one informed, not alphabetically —
so you can find the source behind any claim in the deck.

Roughly 45 sources. The ten marked **★** are the ones that actually changed a
decision; the rest are supporting or context.

---

## 1 · Foundations — the physics our whole design obeys

### ★ Kuo & Morgan — *Active Noise Control: A Tutorial Review*
http://www2.coe.pku.edu.cn/tpic/2010913102917710.pdf

**Says:** the standard reference for active noise control. Defines FxLMS, the
secondary-path problem (the speaker and air change your anti-noise before it
arrives), and the **causality condition**.

**Why we used it:** everything in L0 is built on this.

**How it helped:** the causality condition became our **146 µs budget**, which
then decided the chip, the microphone type and the whole three-layer split. This
is the single most load-bearing source in the project.

### ★ *Causality study on a feedforward ANC headset with different noise directions in free field*
https://opus.lib.uts.edu.au/bitstream/10453/118176/4/Causality%20study%20on%20a%20feedforward%20active%20noise%20control%20headset%20with%20different%20noise%20coming%20directions%20in%20free%20field.pdf

**Says:** whether a headset can cancel at all depends on **which direction the
noise comes from** — the causal window changes with angle.

**Why we used it:** it told us direction is a physical constraint, not a feature.

**How it helped:** produced our **causal cones** result — each cup is causal over
a one-sided angular range, and letting an ear use the *opposite* cup's microphone
lifts coverage **38% → 68%**. That is one of our two novelty claims.

### *A survey on active noise control techniques — Part I: Linear systems*
https://arxiv.org/pdf/2110.00531

**Says:** systematic survey of classical ANC. **Why:** to be sure we were not
reinventing something standard. **How:** confirmed the classical toolbox and its
limits on non-stationary noise.

### *Toward Optimal ANC: Establishing a Mutual Information Lower Bound*
https://arxiv.org/pdf/2505.17877

**Says:** an information-theoretic limit on what any controller can achieve.
**Why:** to know what "good" even means. **How:** supports our use of a
*predictability* measure — if the reference cannot predict the disturbance, no
algorithm helps.

---

## 2 · Impulsive noise — the gunshot problem

### ★ *Review of Advances in Active Impulsive Noise Control* (Applied Sciences, 2024)
https://doi.org/10.3390/app14031218

**Says:** ordinary LMS **diverges** on impulsive noise — one huge sample makes the
filter learn something catastrophically wrong. Survey of robust fixes.

**Why we used it:** gunfire and artillery are the defining case in our problem
statement.

**How it helped:** gave us the robust score function in L0. We then measured that
**normalisation is itself an impulse defence** — the textbook divergence only
reproduces with *un*normalised FxLMS.

### *Improving robustness of filtered-x least mean p-power for α-stable impulsive noise*
https://www.sciencedirect.com/science/article/abs/pii/S0003682X1100051X

**Says:** models impulsive noise as α-stable and gives a p-power update rule that
survives it. **Why:** the concrete algorithm behind the review above. **How:**
our `score()` function.

### ★ MIL-STD-1474E (2015)
https://arl.devcom.army.mil/wp-content/uploads/sites/3/2022/09/ahaah-MIL-STD-1474E-Final-15Apr2015.pdf
— superseding MIL-STD-1474D: https://www.denix.osd.mil/soh/denix-files/sites/21/2022/12/03_MILSTD1474D-Noise-Limits.pdf

**Says:** the US military noise standard **dropped its simple peak-loudness rule**
in favour of a model of how the ear is actually damaged.

**Why we used it:** it is official evidence that impulse noise is unsolved.

**How it helped:** this is our strongest "the problem is real" citation — a
standards body changed its rule because the old one was not protecting people.

### *Firearms and hearing protection*
https://hearingreview.com/hearing-loss/patient-care/evaluation/firearms-and-hearing-protection

**Says:** gunfire sound levels and what protection actually achieves.
**Why/How:** grounds the 160 dB figure we quote.

---

## 3 · The AI-selects-the-filter family — the seed of our L2

### ★ SFANC — *Selective Fixed-Filter ANC based on a CNN* (Signal Processing, 2021)
https://www.sciencedirect.com/science/article/abs/pii/S0165168421003546

**Says:** a CNN **picks**, per frame, the best filter from a bank of pre-trained
fixed filters. Fast response, no per-sample adaptation transient.

**Why we used it:** it shows a network can help ANC **without being in the audio
path**.

**How it helped:** this is the direct ancestor of our **L2** — the AI writes
coefficients at 62.5 Hz and never touches sound. We measured **+9.3 dB** from
exactly this mechanism.

### GFANC — *Generative Fixed-Filter ANC* (ICASSP 2023)
https://arxiv.org/abs/2303.05788

**Says:** *generates* filters by combining sub-band pieces of one broadband
filter, so far less prior data is needed. **Why/How:** shows the filter bank need
not be enumerated in advance — relevant if our noise classes grow.

### Hybrid SFANC-FxNLMS
https://arxiv.org/abs/2208.08082

**Says:** a 1D CNN selects the filter **and** FxNLMS keeps adapting it per sample.
**Why:** this is closest to what we actually built. **How:** validates combining
selection with continuous adaptation rather than choosing one.

### PFANC — predictive filter selection (2026)
https://arxiv.org/pdf/2606.08171

**Says:** predicts the **next** frame's filter, removing selection lag.
**Why/How:** justifies L2 running ahead at 62.5 Hz rather than reacting late.

### ★ D-SFANC / PD-SFANC / SF-GFANC — directional filter selection (2026)
https://arxiv.org/abs/2601.06981 · https://arxiv.org/abs/2604.23144 · https://arxiv.org/abs/2607.12807

**Says:** use the estimated direction of the noise to choose the filter; PD-SFANC
handles a **moving** source on a 36-sector grid.

**Why we used it:** this is our **closest prior art** — we must be able to say how
we differ.

**How it helped, in two ways.** First, it is the honest baseline for our direction
claim: they use bearing to pick a *filter*, we use it to pick a *microphone*.
Second — and more useful — **PD-SFANC reports direction accuracy above 90% only at
SNR ≥ 20 dB.** That is a quiet room, not an armoured vehicle. We designed a
confidence gate so the system falls back to a fixed filter instead of steering
confidently in the wrong direction.

### *Selective fixed-filter ANC by frequency-response matching in headphones* (Applied Acoustics, 2023)
https://www.sciencedirect.com/science/article/abs/pii/S0003682X23003031

**Says:** applies filter selection specifically to **headphones** rather than
ducts or rooms. **Why/How:** confirms the approach transfers to our form factor.

### *Neural network-based ANC algorithms: a review*
https://www.extrica.com/article/25037

**Says:** survey of where neural networks are being used in ANC.
**Why/How:** used to check we had not missed a family of approaches.

---

## 4 · Putting the network *in* the audio path — the approach we rejected

### ★ Deep ANC (Zhang & Wang) — Neural Networks 2021 / Interspeech 2020
https://www.sciencedirect.com/science/article/abs/pii/S0893608021001258 ·
https://www.isca-archive.org/interspeech_2020/zhang20i_interspeech.pdf

**Says:** treat ANC as supervised learning — a network estimates the cancelling
signal directly. **More than 6 dB better than FxLMS** in untrained noise.

**Why we used it:** this is the **strongest argument against our architecture**,
so we have to be able to answer it.

**How it helped:** it sharpened our claim. Deep ANC is better *if* you can run a
network inside the 146 µs budget. On a ₹409 chip you cannot — we measured the
compute. So we put the intelligence where the deadline is loose instead. Knowing
this paper is what makes our answer credible rather than defensive.

### *Deep Active Speech Cancellation with a Mamba-masking network* (2026)
https://arxiv.org/abs/2502.01185

**Says:** masking on the encoded reference plus multi-band phase alignment, +7.2 dB.
**Why/How:** the current state of the art in the in-path approach — same trade-off,
same verdict for us.

### *Speech-preserving deep ANC in reverberant environments* (2026)
https://arxiv.org/abs/2604.10979

**Says:** adds a **speech-retention loss** that suppresses noise while *keeping*
the target voice. **Why:** exactly our failure mode. **How:** shaped our
asymmetric loss, which penalises deleting speech 8× harder than leaving noise.

### DecNet-LMS (2026)
https://arxiv.org/pdf/2511.03162

**Says:** a fixed-weight network inverts the secondary path while an LMS filter
models the primary path online, in the time domain to preserve phase.
**Why/How:** another hybrid split of duties; supports our "network outside the
sample loop" instinct.

---

## 5 · The small speech-enhancement models our L1 is measured against

### ★ GTCRN (ICASSP 2024) — **48.2 k parameters, 33.0 MMAC/s**
https://ieeexplore.ieee.org/document/10448310/ · code: https://github.com/Xiaobin-Rong/gtcrn

**Says:** a speech enhancer needing ultra-low compute, using grouped convolutions,
sub-band features and a dual-path RNN.

**Why we used it:** it is the efficiency reference point for the whole project.

**How it helped:** our L1 is scaled against it — **22,956 parameters, half its
size**. It is also why we rejected pruning: there is nothing left to squeeze.

### LiSenNet (ICASSP 2025) — **37 k parameters**
https://ieeexplore.ieee.org/document/10888272/

**Says:** sub-band modelling plus a noise detector that **skips computation** on
frames with no speech. **Why/How:** the source of our conditional-computation idea
and the no-op gate for high-SNR frames.

### DCCRN (Interspeech 2020)
https://arxiv.org/pdf/2008.00264

**Says:** heavy suppression without repairing **phase** produces metallic speech
and destroys stop consonants. **Why:** fatal for call signs and coordinates.
**How:** it is why we built a phase branch at all — and, ironically, our own
measurement later showed our *band-level* phase rotation does more harm than
good. We report both.

### GTFCRN — grouped iterative CRN, INT8 on a CSK6012
https://link.springer.com/article/10.1186/s13636-026-00455-4

**Says:** a low-parameter enhancer actually deployed in INT8 on a
noise-reducing-headphone chip. **Why/How:** proof the class of model we chose does
reach real hardware.

### *A Lightweight Hybrid Dual-Channel Speech Enhancement System under Low-SNR* (Interspeech 2025)
https://arxiv.org/abs/2505.19597

**Says:** two-microphone enhancement tuned for low SNR. **Why/How:** low SNR is
exactly where our STOI fails, so this is a live candidate for future work.

---

## 6 · Loss functions and training — where our current problem sits

### ★ Wang, Zhu & Kodrasi — *Speech-presence probability as a secondary task* (ICASSP 2021)
https://arxiv.org/abs/2011.07547

**Says:** train the network to also predict *where speech is*, weighting the two
tasks by uncertainty.

**Why we used it:** free regularisation — the shared encoder learns "where is
speech" instead of only "what to output".

**How it helped:** it is our **SPP head**, and it specifically protects weak
consonants, which are the first thing lost in heavy suppression.

### PercepNet+ — phase and SNR aware PercepNet (Interspeech 2022)
https://arxiv.org/pdf/2203.02263 · Personalized PercepNet: https://www.isca-archive.org/interspeech_2021/giri21_interspeech.pdf

**Says:** per-band gains plus a **pitch-driven comb filter** that removes noise
*between* voice harmonics, at very low cost — PESQ-WB 2.54 at under 5.2% of a
CPU core.

**Why we used it:** it looked like the direct fix for our PESQ deficit.

**How it helped — by failing.** We implemented it and measured **+0.004 PESQ**:
nothing. That null result is what proved our problem is *distortion*, not
leftover noise, and it redirected the entire project. A source that changed our
direction by being wrong for us.

### Learnable comb filter for lightweight full-band SE (2023)
https://arxiv.org/abs/2306.00812

**Says:** classical pitch trackers mis-estimate F0 and the comb filter then
*degrades* speech; fix it by learning F0 end-to-end. **Why/How:** the mitigation
for the one real failure mode of PercepNet, had we pursued it.

### HC-APNet — harmonic compensation for low-complexity SE (2024)
https://www.sciencedirect.com/science/article/abs/pii/S0167639324001328

**Says:** independent confirmation of harmonic compensation at low cost.
**Why/How:** cross-check on the family above.

---

## 7 · 2025–26 work on hallucination — why we will not use generative models

### ★ *A Comparison of Generative and Discriminative Methods for Speech Enhancement: Robustness, Complexity, and Hallucination*
https://arxiv.org/html/2606.02913

**Says:** discriminative models are efficient but "over-smoothed or distorted";
generative models sound best but **fabricate content**, and hallucination is
**much worse at low SNR**. Also: DNSMOS scores stay high *even when content is
hallucinated*.

**Why we used it:** a defence radio that invents a word is a safety failure.

**How it helped:** it is the citation behind our hard rule — **the output must
remain a bounded filter of the input**, so fabrication is structurally impossible.
It also warned us not to trust non-intrusive metrics alone.

### PASE / UniPASE — low-hallucination generative SE
https://arxiv.org/pdf/2511.13300 · https://arxiv.org/pdf/2604.14606

**Says:** attempts to reduce hallucination *inside* generative enhancement.
**Why/How:** named and **rejected on purpose** — still generative, still far too
heavy for an MCU. Being able to say why is worth a slide.

### *Reducing Linguistic Hallucination in LM-Based Speech Enhancement*
https://arxiv.org/html/2605.08608

**Says:** distillation to stop language-model enhancers inventing words.
**Why/How:** same family, same rejection.

### Interspeech 2025 URGENT Speech Enhancement Challenge
https://arxiv.org/pdf/2505.23212

**Says:** current community benchmark and its findings on robustness.
**Why/How:** context for where the field is, and what evaluation is considered
credible.

---

## 8 · Embedded deployment

### ★ Espressif ESP-DSP benchmarks
https://docs.espressif.com/projects/esp-dsp/en/latest/esp32/esp-dsp-benchmarks.html

**Says:** published CPU cycle counts for optimised DSP routines. `dsps_dotprod_s16`
does 256 multiply-accumulates in **307 cycles**.

**Why we used it:** to decide whether the whole system fits on a ₹409 chip using
the vendor's own measurements rather than our estimates.

**How it helped:** gave us **200 MMAC/s per core**, which produced the full timing
budget — **259 of 400 MMAC/s, 65%, frame deadline PASS** — and the L0 filter
length decision.

### *Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with mixed FP16-INT8 post-training quantization*
https://arxiv.org/abs/2210.07692

**Says:** how to actually get a recurrent enhancer onto a multi-core
microcontroller. **Why/How:** our deployment template, and a warning that RNNs
are the hard part — which our own profiling confirmed (**76% of training time is
the GRU**).

### Real-time SE at 3.35 ms end-to-end, 4.1 dB SI-SDRi, 376 MIPS on a low-power DSP
https://arxiv.org/pdf/2409.18239

**Says:** a working low-latency deployment with published numbers.
**Why/How:** a sanity check that our latency and MIPS targets are realistic.

### *Feasibility of Time-Domain DNN Speech Enhancement on Embedded FPGA for Hearing Aids*
https://arxiv.org/pdf/2606.04221

**Says:** the FPGA alternative to an MCU. **Why/How:** considered and set aside on
cost and toolchain maturity for a student team.

### ADAU1772 datasheet (Analog Devices)
https://www.analog.com/media/en/technical-documentation/data-sheets/ADAU1772.pdf

**Says:** 4 ADC / 2 DAC low-power codec; **group delay 38 µs at fS = 192 kHz**;
output configurable as line or headphone driver; digital supply from an on-chip
regulator; absolute max AVDD/IOVDD **3.63 V**.

**Why we used it:** this is the deciding part of the whole BOM.

**How it helped:** every hardware decision traces here — no level shifters, no
amplifier IC, a single 3.3 V rail, and the 4-ADC limit that fixed our microphone
count at 2 reference + 2 error.

---

## 9 · Hearing, safety and the human case

### *Impact of noise on hearing in the military*
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4455974/

**Says:** prevalence and mechanism of noise-induced hearing loss in service
personnel. **Why/How:** the evidence base for our impact slide.

### *Noise and Noise-Induced Hearing Loss in the Military* — National Academies
https://www.nationalacademies.org/read/11443/chapter/5

**Says:** 21% noise-induced hearing loss and 28% tinnitus in post-deployment
records; hearing damage is the **second most prevalent** service-connected
disability. **Why/How:** turns "soldiers should hear better" into a measured,
costed problem.

### CDC / NIOSH hearing protection guidance
https://stacks.cdc.gov/view/cdc/111710/cdc_111710_DS1.pdf

**Says:** occupational hearing-protection standards. **Why/How:** the civilian
market case — mining, aviation ground crew, heavy industry.

### *Restoring Spatial Hearing in Active Noise Control Headphones* (DAGA 2025)
https://pub.dega-akustik.de/DAS-DAGA_2025/files/upload/paper/96.pdf

**Says:** cancelling noise must not cost you the ability to tell **where** a sound
came from. **Why/How:** a safety constraint, not a feature — a soldier who cannot
localise a shot is worse off. It is why we treat spatial cues as something to
preserve.

### *Digital Augmented Reality Audio Headset* — hear-through latency
https://onlinelibrary.wiley.com/doi/10.1155/2012/457374

**Says:** how much delay is tolerable when passing outside sound through to the
ear. **Why/How:** bounds the hear-through path.

---

## 10 · Datasets

| Source | What it gives | Licence | Link |
|---|---|---|---|
| **LibriSpeech** | Clean speech, speaker-disjoint splits | CC BY 4.0 | https://www.openslr.org/12 |
| **MUSAN** | Environmental and background noise | per-file CC / public domain | https://www.openslr.org/17 |
| **MAD** — Military Audio Dataset | Gunshot, shelling, fighter, helicopter, footsteps | CC BY 4.0 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11193796/ |
| **RIRS_NOISES** | Real measured room impulse responses | Apache 2.0 | https://www.openslr.org/28 |
| **sireNNet** | Police, fire, ambulance sirens | usable, partly augmented | Kaggle |
| **Drone corpus** | Rotor / UAV noise | usable | Hugging Face |
| **DNS Challenge** | Clean speech backbone, noise variety | usable | https://github.com/microsoft/DNS-Challenge · https://arxiv.org/pdf/2005.13981 |
| **NOISEX-92** | Classic noise benchmark | reference only | http://mi.eng.cam.ac.uk/comp.speech/Section1/Data/noisex.html |

**Deliberately excluded, and this matters in the Q&A:**

- **ESC-50** and **UrbanSound8K** — CC BY-**NC**, non-commercial only. A defence
  product is commercial.
- **DEMAND** — CC BY-**SA**, share-alike would propagate to our work.
- **MAD's `communication` class** — contains speech. Speech in the noise pool
  teaches the model to delete voices.

---

## 11 · Market and policy sources

| Source | What it gives | Link |
|---|---|---|
| **PIB, Ministry of Defence** | India MoD budget FY 2025–26: ₹6,81,000 cr total, ₹1,80,000 cr capital, **75% of modernisation reserved for domestic sources** | https://www.pib.gov.in/PressReleasePage.aspx?PRID=2098485 |
| **MP-IDSA** | Independent analysis of MoD budget estimates | https://idsa.in/publisher/issuebrief/ministry-of-defence-2026-27-budget-estimates-an-analysis |
| **Fortune Business Insights** | India defence electronics: $6.85 B (2024) → $11.35 B (2032) | https://www.fortunebusinessinsights.com/india-defense-electronics-market-114279 |
| **Fortune Business Insights** | Tactical headset market | https://www.fortunebusinessinsights.com/tactical-headset-market-108385 |
| **Verified Market Reports** | Tactical communication headsets: $3.64 B (2026) → $5.81 B (2033), 7% CAGR | https://www.verifiedmarketreports.com/product/tactical-communication-headset-market/ |

**Say "projected to" when quoting these.** They are third-party forecasts, not
measurements. Only RHEAR's own figures come from code we ran.

---

## The ten that actually changed a decision

1. **Kuo & Morgan** → the 146 µs causality budget → the entire architecture
2. **Causality study (UTS)** → causal cones → 38% → 68% coverage
3. **SFANC (2021)** → the AI writes coefficients, never audio → our L2
4. **PD-SFANC (2026)** → its SNR ≥ 20 dB limitation → our confidence gate
5. **Deep ANC (2021)** → the strongest counter-argument → forced us to justify our split
6. **GTCRN (2024)** → the efficiency target → and why pruning is pointless for us
7. **Wang, Zhu & Kodrasi (2021)** → our speech-presence head
8. **PercepNet (2020)** → we implemented it, it gave +0.004 → **found the real problem**
9. **Generative vs Discriminative (2026)** → our bounded-mask, no-hallucination rule
10. **ESP-DSP benchmarks** → 200 MMAC/s/core → the timing budget and the ₹409 chip

**MIL-STD-1474E** deserves an honourable mention: it is not a method, but it is
the best evidence that the problem is real and officially unsolved.
