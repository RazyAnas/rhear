# Codebase map and how to run things

Root: `~/PS#2` (note the `#` — quote paths in shell: `cd ~/PS\#2`)

## The digital twin (the demo)

```
demo.sh                     one-command launcher -> http://127.0.0.1:8765
rhear_twin.py               the twin server
rhear/
  core/anc.py               FxNLMS, score functions
  core/l0.py                L0 runtime
  core/l2runtime.py         the coefficient interface -- the architecture's core
  core/predict.py           prediction floor (LEVINSON-DURBIN, see gotcha 11)
  core/doa.py               direction estimation
  core/signals.py           generators
  sim/                      geometry, acoustic paths
  telemetry/schema.py       FLOW_NODES / FLOW_EDGES / COEFFICIENT_EDGES
  telemetry/server.py       HTTP + SSE, serves /api/ab and /ab/*.wav
  telemetry/sources/simulation.py   the live simulation source
  experiments/e06_filter_generation.py   THE L2 CLAIM (+9.3 dB)
  experiments/e07_structural.py, e08_filter_length.py
ui/index.html               the dashboard, no framework, no build
```

Start it: `cd ~/PS#2 && ./demo.sh` — **takes ~15 s** to train the filter bank.

## L1 (the neural enhancer) — everything under `E03/`

| file | purpose |
|---|---|
| `model/gtcrn_lite.py` | **the model.** `GTCRNLite(ch, phase, fullband, df, hop)` |
| `train_interim.py` | training. Also holds `enhance()`, `stft/istft`, `loss_fn` |
| `eval_stratified.py` | **the evaluator.** `load_model()` infers variant from the checkpoint |
| `crossbench.py` | **runs GTCRN's published weights on our data** — the key result |
| `oracle_ladder.py` | ceiling of every candidate design, no training |
| `phase5_sar_ceiling.py` | the original oracle probe |
| `build_demo_ab.py` | rebuilds the demo A/B pack from a checkpoint |
| `model_diff.py` | params / MACs / ONNX operators / Flash / RAM |
| `bench/timing_budget.py` | the whole-headset MMAC/s budget |
| `check_leakage.py` | **run on any new dataset build** |
| `probe_variant.py` | tells G0 from G1 by loss level |
| `run_after_retrain.sh` | the six agreed post-training steps |

## Common commands

```bash
cd ~/PS#2/E03

# train (LAPTOP: always --no-cache, see gotcha 14)
python3 -u train_interim.py --data ../handoff/data/h3_20k \
    --epochs 24 --batch 32 --no-cache --lr 2e-3 \
    --hop 256 --ch 32,48,48,64 --no-phase --fullband \
    --out runs/NAME

# evaluate on both sets
for S in edef realnoise; do
  python3 eval_stratified.py --data ../handoff/data/$S \
      --ckpt runs/NAME/best.pt --out runs/g012/eval --tag NAME_on_$S
done

# the cross-benchmark (3 min, no training)
python3 crossbench.py --ckpt dns3

# ceilings
python3 oracle_ladder.py --n 40 --window 24

# resources + budget
python3 model_diff.py && python3 bench/timing_budget.py

# rebuild the demo audio from a checkpoint
python3 build_demo_ab.py --ckpt runs/NAME/best.pt
```

## Training flags that matter

| flag | meaning |
|---|---|
| `--no-phase` | drop the learned phase branch. **Always use** — measured harmful |
| `--fullband` | G2's parallel full-band branch. **Keep** |
| `--df` | G6 deep-filtering second stage |
| `--hop` | 64 = 250 fps (historic), 256 = 62.5 fps (GTCRN's operating point) |
| `--ch` | channel widths, e.g. `32,48,48,64` = 49,663 params |
| `--init` | fine-tune from a checkpoint; reports what did and did not load |
| `--no-cache` | **required on the laptop** |
| `--perceptual` | rejected at 1.0 and 0.15 |

## Checkpoints

`E03/runs/*/best.pt` (best val SI-SDR) and `last.pt` (final epoch).
**For anything targeting PESQ, evaluate BOTH** — `best.pt` is selected on
SI-SDR and is biased against perceptual interventions (G5 selected epoch 0 of 4
for exactly this reason).

Key runs: `h3_20k` (frozen baseline), `g2_fullband` (**current best**),
`g6_deepfilter2`, `g7_hop256_50k` (in flight), `h9_frozen` (frozen H3 snapshot).

## Data

`~/PS#2/handoff/data/` — `h3_20k` (20k training mixtures), `edef` (defence eval,
300 test), `realnoise` (environmental eval, 300 test). 2.6 GB.

## Docs

`docs/06-L1-artefact-evidence.md` is **the most important document** — every
experiment, including all the negative results.
Also: `04b-bom-verified.md` (hardware), `05-digital-twin.md`,
`PRESENTATION-SCRIPTS.md` + PDF (six speaker scripts), `REFERENCES.md` + PDF,
`DEMO-GUIDE.md` + PDF. Convert MD→PDF with `python3 docs/md2pdf.py IN.md OUT.pdf`.
