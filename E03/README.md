# RHEAR E03 — dataset and L1 baseline

**Status: dataset pipeline built, model specified and measured, NOTHING TRAINED.**

| | |
|---|---|
| `RHEAR_E03_dataset.ipynb` | Colab notebook: download → build → audit → DATASET REPORT |
| `rhear_data/registry.py` | dataset registry with **verified** licences and the exclusion policy |
| `rhear_data/mixing.py` | mixture engine: real speech + real noise, SNR and level trajectories |
| `rhear_data/headset.py` | headset realism: mic response, self-noise, clipping, RIR, level mismatch |
| `rhear_data/splits.py` | speaker / noise-source / room disjoint splits + leakage audit |
| `rhear_data/manifest.py` | per-sample provenance, seeds, library versions |
| `rhear_data/report.py` | the DATASET REPORT, computed not estimated |
| `model/gtcrn_lite.py` | the L1 baseline — run it to print params, MACs, latency |
| `model_spec.md` | architecture, measured budgets, loss, runtime estimate |
| `tests/` | causality, budget and mask-bound tests |

## Provenance vocabulary — used strictly

| tag | meaning |
|---|---|
| `real recording` | unmodified audio from a public corpus |
| `synthetically mixed from real recordings` | real speech + real noise combined by this pipeline — **this is what the RHEAR dataset is** |
| `synthetically generated` | produced by a signal generator — **not used in E03** |

## Licence policy

Default build uses only CC BY 4.0 / CC0 / US Public Domain / Apache 2.0.
**ESC-50 and UrbanSound8K are excluded** (CC BY-NC — non-commercial).
**DEMAND is excluded** (CC BY-SA — share-alike is viral).
Derived dataset licence: **CC BY 4.0 with attribution to each source corpus.**

**MAD is used as a military-noise source only** — it is a classification corpus
with no clean-speech pairing, and its `communication` class is excluded from the
noise pool because it contains speech.

## Run

```bash
python3 model/gtcrn_lite.py            # model budgets
python3 tests/test_model_causality.py  # causality + budgets + mask bound
```

The dataset itself is built in Colab — see the notebook. It needs ~12 GB
(small profile) cached to Drive, and is reproducible from a fresh session.
