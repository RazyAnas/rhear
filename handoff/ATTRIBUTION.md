# Attribution
The audio in `data/` is **derived** from the corpora below. Its derived
licence is **CC BY 4.0 with attribution to each source corpus**, so redistribution
carries an attribution requirement. This file exists to satisfy it.

Every mixture's full provenance is recorded per-clip in the `manifest.jsonl`
of each dataset folder, including which source recording each noise layer
came from and whether it is a real recording, a real-but-augmented one, or
synthetic.

| Source corpus | Licence | Content |
|---|---|---|
| LibriSpeech dev-clean | CC BY 4.0 | real recording |
| LibriSpeech test-clean | CC BY 4.0 | real recording |
| MUSAN (noise subset) | per-file CC or US Public Domain | real recording |
| MAD - Military Audio Dataset | CC BY 4.0 | real recording |
| RIRS_NOISES (OpenSLR 28) | Apache 2.0 | real recording (real RIR subset) + synthetic (simulated subset) |

## Deliberately excluded

- **ESC-50** and **UrbanSound8K** — CC BY-NC, non-commercial only.
- **DEMAND** — CC BY-SA, share-alike would propagate to this work.
- **MAD `communication` class** — contains speech; including it in a noise
  pool would train the model to delete voices.
