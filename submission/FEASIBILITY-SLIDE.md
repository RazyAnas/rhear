# FEASIBILITY & VIABILITY — one slide

## ► USE THIS ONE (fits the measured slide budget)

### FEASIBILITY

| | Evidence | |
|---|---|---|
| **Technical** | 97.4 KB int8 of a 200 KB cap; ~15% of one core. Model trained and benchmarked; neural VAD running on-chip; FxNLMS validated +12.6 dB | **Proven** |
| **Financial** | India-sourced BOM ₹6,370; **₹4,071 as retrofit**. ~₹20,000 fielded vs **₹1.8 lakh** for the reference system. iDEX / TDF are funded routes | **Costed** |
| **Market** | **22.9%** of IAF personnel have hearing loss (**26.18%** technical trades). Reference system reached ~20,000 worldwide — price cited as the limit | **Unmet** |
| **Operational** | Retrofits into an in-service earmuff: no new shell, no re-qualification. All parts from Indian distributors, **none export-controlled** | **Low friction** |

### VIABILITY — challenges and answers

| Challenge | Answer | Status |
|---|---|---|
| Quality short at worst noise levels | We measured the ceiling — a *perfect* single-mic mask can't reach it. Second mic separates wearer from bystander **100%**, +10.1 dB | Simulated, 2,160 scenes |
| Model not yet fully on-chip | Memory and compute are not the limit. Largest block validated to **−133.5 dB** vs PyTorch | Phase 1, in progress |
| Cancellation needs low-latency codec | 146 µs budget is geometry. Codec chosen on 38 µs; circuit **ERC-clean, 37 parts** | Designed |
| Ruggedisation / qualification | Phase 4. Certified passive shell unchanged — only electronics need qualifying | Planned |
| Siren data thin | Loader already in codebase; a rebuild, not research. **We don't claim what we haven't measured** | Scoped |

**Every item was found by our own measurement, has a costed answer, and is scheduled.**

---

## ► FULLER VERSION (if your template allows more text)

## TABLE 1 — FEASIBILITY

| | Evidence | Verdict |
|---|---|---|
| **Technical** | 97.4 KB int8 vs the 200 KB cap; 26–32 MMAC/s vs ~200 available per core (~15%). Model trained and benchmarked; neural VAD already running on the ESP32-S3; FxNLMS validated at +12.6 dB. | **Proven** |
| **Financial** | Verified India-sourced BOM ₹6,370 complete, **₹4,071 as a retrofit**. At 5× BOM for enclosure and qualification, ~₹20,000 fielded vs **₹1.8 lakh** for the reference system. iDEX and TDF (to ₹50 cr) are funded routes. | **Costed** |
| **Market** | **22.9%** of IAF personnel show noise-induced hearing loss, **26.18%** in technical trades. The reference system reached only ~20,000 soldiers worldwide, its price openly cited as the limit. Defence production ₹1.78 lakh cr FY26 → ₹3 lakh cr by 2029. | **Unmet need** |
| **Operational** | Retrofits into a 3M Peltor X3A already in service — no new shell, no re-qualification of certified passive attenuation, no change in how it is worn. Every BOM line available from Indian distributors; **no export-controlled part**, so no licence step. | **Low friction** |

## TABLE 2 — VIABILITY: WHAT COULD GO WRONG, AND THE ANSWER

| Challenge | Our answer | Status |
|---|---|---|
| Quality target not met at the hardest noise levels | We measured the ceiling: a *perfect* single-mic mask cannot reach it. A second boom mic separates wearer from bystander at **100%**, +10.1 dB, using wavefront curvature — no enrolment | Simulated over 2,160 scenes |
| Full neural model not yet on the microcontroller | Memory and compute are **not** the constraint (97 KB of 200; 15% of a core). Largest block — the bidirectional GRU — already ported and validated to **−133.5 dB** vs PyTorch | In progress, Phase 1 |
| Active cancellation needs a low-latency codec | 146 µs causality budget is geometry, not opinion. Codec selected on its 38 µs datasheet figure; circuit complete and **ERC-clean, 37 parts / 30 nets** | Designed, awaiting part |
| Ruggedisation, environmental qualification | Deliberately Phase 4. Retrofit scope means the certified passive shell is unchanged, so qualification is of the electronics only | Planned |
| Siren class thin in training data | Loader already in the codebase from an earlier build; restoring it is a dataset rebuild, not new research. **We do not claim siren performance we have not measured** | Known, scoped |

## ONE LINE UNDER THE TABLES

Nothing above is a blocker we discovered late — each was found by our own
measurement, has a costed answer, and is scheduled.
