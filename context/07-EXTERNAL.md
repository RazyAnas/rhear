# Datasets, licences, papers, links

## Licence policy — CHANGED, read this

Originally we excluded non-commercial datasets. **The user has explicitly
relaxed this**: *"we can use even non commercial datas also because its just
project not for commercial use just a demo that yeah we are capable of doing
so."*

| dataset | licence | status |
|---|---|---|
| LibriSpeech | CC BY 4.0 | in use (clean speech) |
| MUSAN | CC BY 4.0 | in use (noise) |
| MAD | — | in use. **`communication` class contains SPEECH — never put it in the noise pool** |
| RIRS_NOISES | — | in use (real measured reverberation) |
| **ESC-50** | CC BY-NC | **now permitted** — was excluded |
| **UrbanSound8K** | CC BY-NC | **now permitted** — was excluded |
| **DEMAND** | CC BY-SA | **now permitted** — was excluded (share-alike) |

The last three are the Track B expansion and are not yet integrated.
Derived audio in `E03/runs/demo_ab/` is CC BY 4.0 with attribution to each
source corpus.

## Our datasets

- `handoff/data/h3_20k` — 20,000 training mixtures, SNR −10..+20 dB, 9 noise
  classes, 12,080 of 20,300 with real measured reverberation, 3,091 clipped
- `handoff/data/edef` — 300 test clips, 8 defence classes, 177 sources, 93 RIRs
- `handoff/data/realnoise` — 300 test clips, environmental

Sources verified disjoint from training. **Run `check_leakage.py` on any rebuild.**

## The papers that matter

**Our L2 — AI selects the filter, never the audio**
- SFANC, *Selective Fixed-Filter ANC based on a CNN*, Signal Processing 2021 —
  https://www.sciencedirect.com/science/article/abs/pii/S0165168421003546
- GFANC, ICASSP 2023 — https://arxiv.org/abs/2303.05788
- Deep ANC (Zhang & Wang), Neural Networks 2021 — **the approach we rejected**,
  network inside the audio path —
  https://www.sciencedirect.com/science/article/abs/pii/S0893608021001258

**The physics behind L0**
- Kuo & Morgan, *Active Noise Control: A Tutorial Review*, Proc. IEEE 1999 —
  http://www2.coe.pku.edu.cn/tpic/2010913102917710.pdf
- *Causality study on a feedforward ANC headset with different noise directions
  in free field* — our 146 us budget and 7 cm geometry come from here —
  https://opus.lib.uts.edu.au/bitstream/10453/118176/4/Causality%20study%20on%20a%20feedforward%20active%20noise%20control%20headset%20with%20different%20noise%20coming%20directions%20in%20free%20field.pdf
- ADAU1772 datasheet (the 38 us) —
  https://www.analog.com/media/en/technical-documentation/data-sheets/ADAU1772.pdf

**Our L1 — low-complexity speech enhancement**
- **GTCRN**, ICASSP 2024 — 48,245 params, PESQ 2.87 on VoiceBank+DEMAND.
  **We run their weights on our data in `crossbench.py`.**
  https://github.com/Xiaobin-Rong/gtcrn · https://ieeexplore.ieee.org/document/10448310/
- **DeepFilterNet**, ICASSP 2022 — the deep-filtering second stage, ~+0.2 PESQ
  over ERB gains alone — https://arxiv.org/abs/2110.05588
- **DeepFilterNet2**, IWAENC 2022 — N=5 taps, f_DF = 5 kHz, 32 ERB bands —
  https://arxiv.org/abs/2205.05474
- **PercepNet** (Valin), Interspeech 2020 — 34 ERB bands, <5% of a CPU core —
  https://arxiv.org/abs/2008.04259
- **LiSenNet** — 37k params, PESQ 2.95–3.08 — https://arxiv.org/html/2409.13285

Full annotated list with what each says and why we used it:
`docs/REFERENCES.md` (+ PDF, 9 pages, 61 links).

## Our published artefacts

- **GitHub:** https://github.com/RazyAnas/rhead — the demo, runnable with
  `git clone`, `pip install numpy scipy psutil`, `./demo.sh`.
  **NOTE: the repo copy still has the pre-G2 `summary.json`** — the local
  `E03/runs/demo_ab/` is newer. Push it when convenient.
- **Architecture diagram artifact:**
  https://claude.ai/code/artifact/ce85e0df-3cf6-4155-8353-4d1bc8c1eb45
  (corrected signal architecture, colour-coded, with the ESP32-S3 marked)

## Security

**A Kaggle credential (`KGAT_11b3f6de...`) was pasted into a chat transcript and
written to `~/.kaggle/access_token`. IT MUST BE ROTATED.**
