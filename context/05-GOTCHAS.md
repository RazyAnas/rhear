# Traps. Read before touching code.

Each of these cost real hours.

## Data and evaluation

1. **The leakage guard exists for a reason.** `E03/check_leakage.py`. `build()`
   emits throwaway 1-clip `train`/`val` rows beside the real `test` split
   because `n_per_split` requires all three. Those dummies draw from the whole
   pool and WILL collide with training sets. They are never evaluated.
   **Compare the TEST split only.** A false leakage alarm from including them
   cost an afternoon and nearly triggered a pointless dataset rebuild.
2. **The MAD `communication` class contains speech.** It must never enter the
   noise pool.
3. **Exclusion sets must be applied to every index**, not just `env_index` —
   727 E_def recordings once leaked into H3's pool because only one index was
   filtered. An assert caught it.
4. **Manifest paths say `.wav`; files may be `.flac`.** The resulting
   `LibsndfileError` gets mangled by DataLoader workers into what looks like a
   fork-safety bug.

## The model contract

5. **`hop` travels inside the checkpoint** as a registered buffer (`hop_`).
   A model trained at hop 256 and evaluated at hop 64 produces plausible-looking
   garbage **with no error**. `eval_stratified.load_model` reads hop AND channel
   widths from the weights. `oracle_ladder.py` asserts the hop matches.
   Checkpoints from before this change have no `hop_` and are injected with 64.
6. **`enhance()` must use the model's hop**, and the training loop's reference
   STFT must use the same one. A mismatch gave 1001 frames against 251.
7. **Parameter count cannot distinguish G0 from G1.** `--perceptual` changes the
   loss, not the model — both are 22,956. Check the `perc` value in the log, or
   epoch-0 `train_loss` (~+16 with, ~−2 without).

## Numerics

8. **`butter()` in `(b,a)` form at 192 kHz puts a pole at radius 1.0055** and
   produces all-NaN. Use SOS.
9. **Power-law compression `|S|^c` diverges as bins approach zero.** `eps` is
   1e-4 on purpose; 1e-8 gave 27.6% non-finite batches.
10. **The perceptual loss must be RMS-normalised against the reference.**
    Unnormalised, a near-silent clip gave term 102.5 vs 5.1 and gradient 6481 vs
    64 — a 100x spike.
11. **`prediction_floor_db` must use Levinson-Durbin.** A generic solver is
    O(n^3) = 350 MMAC/s and the whole budget FAILS.

## Embedded / export

12. **Choose shapes so the arithmetic lands exactly.** G2's full-band strides are
    (4,4,3) — not (4,4,4) — so 257 bins reach exactly 6 without a `Resize`,
    which CMSIS-NN cannot run. The deep-filter head uses `df_bins=183` (not a
    round 192) because a transposed conv gives `(in-1)*stride + kernel`: from 6
    the natural chain at strides (4,4,2) is 6→23→91→183. Asking for 192 made
    each block short by one and inserted `Pad`.
13. **Seven operators are not CMSIS-NN native**: Abs, ConvTranspose, Einsum,
    GRU, LayerNormalization, Pow, Sqrt. Adding an eighth is a real cost.

## Training on this laptop (16 GB Mac)

14. **Never use `--cache-int16` here.** 20k pairs is ~5 GB; it gets paged out,
    the process enters uninterruptible I/O wait, and epochs go 26 → 29 → 50+ min.
    Use `--no-cache`. The flag is correct on the friend's GPU box, wrong here.
15. **Kill the demo twin before training.** `rhear_twin.py` burns ~70% CPU
    continuously. Leaving it running competed with a training run for two hours.
16. **A run getting monotonically slower is a resource problem**, not thermal.
17. **Never pipe training through `tail`** — it buffers everything until exit and
    you cannot read progress. The script now prints **every** epoch (it used to
    print every 5th, which hid epochs 1–3 of a 5-epoch run behind an empty log
    and caused three wrong progress estimates).

## UI / demo

18. **`ui/index.html` reads `stoi_noisy` / `stoi_enhanced`**, not `stoi_n` /
    `stoi_e`. Renaming them made `overall` undefined, the A/B panel stopped
    rendering, and it looked exactly like the audio files had been deleted.
    Key names are a UI contract.
19. **`.gitignore` had a blanket `runs/`** which silently ate
    `E03/runs/demo_ab/*.wav` — the demo's own audio. A fresh clone served the
    dashboard and 404'd every clip. **Always verify by cloning from the remote
    and running it**, never from the directory you committed from.
20. **The twin takes ~15 s to start** — it trains one control filter per scene
    before serving. `demo.sh` waits and prints LIVE; running `rhear_twin.py`
    directly looks like a hang.

## Environment

21. **macOS TCC blocks shell access to `~/Downloads`** in some contexts.
22. **`timeout` is not available** on this macOS shell.
23. **A Kaggle credential (`KGAT_11b3f6de...`) was pasted into a chat transcript
    and written to `~/.kaggle/access_token`. IT MUST BE ROTATED.**
