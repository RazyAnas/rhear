"""RHEAR G7 fine-tune dataset -- FULL-RANGE SNR variant.

This is your make_g7_finetune_ms_snsd.py with exactly ONE thing changed:
the SNR range. Your original hardcoded

    b.SNR_MIN = 0.0
    b.SNR_MAX = 0.0

which trains on a single fixed condition. H3 -- the data everything else in
this project was trained and evaluated on -- spans -10..+20 dB, and
make_g7_realnoise.py's OWN default (b.SNR_MIN / b.SNR_MAX before you override
it) is already -10.0 / 20.0. You were overriding a correct default down to a
single point.

Every other choice is identical to your run: same 70% MAD+DEMAND / 30% MS-SNSD
mix, same CLIP_PROB/DISTORT_PROB, same COUNT, same speaker-leakage checks, same
seed. This writes to a NEW output folder so your existing g7_finetune_ms_snsd_0db
dataset is untouched -- nothing is deleted or overwritten.

Why this matters (measured, not guessed): your 0dB-only checkpoint was
evaluated against our G7-base on our standard held-out sets. SI-SIR went UP
(+0.45 / +1.05 dB -- it suppresses more) but SI-SAR went DOWN HARD
(-2.27 / -0.92 dB -- it damages more), and PESQ fell 0.05-0.11 on both. That is
the signature of a model trained to always suppress hard, because 0 dB is the
only condition it ever saw. A model that only ever sees the loudest, most
adversarial case for the noise learns to always fight the noise -- it never
learns to leave quiet or borderline speech alone.

Run this exactly as you ran the original:
    python3 make_g7_finetune_fullrange.py
Then:
    python3 add_g7_finetune_val_fullrange.py
"""

from pathlib import Path
import sys
import json
import random

# ============================================================
# Paths
# ============================================================

E03_DIR = Path(__file__).resolve().parent
REPO_ROOT = E03_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

import make_g7_realnoise as b


# ============================================================
# Configuration
# ============================================================

OUTPUT_ROOT = (
    b.G7_DATASET
    / "PROCESSED"
    / "g7_finetune_ms_snsd_fullrange"          # NEW name -- your 0db run is untouched
)

COUNT = 10_000
SEED = 45                                       # same seed as your original

MS_SNSD_ROOT = (
    REPO_ROOT.parent
    / "MS-SNSD-master"
    / "MS-SNSD-master"
)

# THE ONE VARIABLE. Explicit values (not "leave b's default alone") so this
# stays correct even if make_g7_realnoise.py's own default ever changes.
b.SNR_MIN = -10.0
b.SNR_MAX = 20.0

# Same augmentation as original training and as your 0dB run -- unchanged
b.CLIP_PROB = 0.15
b.DISTORT_PROB = 0.10


# ============================================================
# Safety
# ============================================================

if OUTPUT_ROOT.exists():
    raise RuntimeError(
        f"\nOutput already exists:\n{OUTPUT_ROOT}\n"
        "Refusing to overwrite it."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

b.OUTPUT_ROOT = OUTPUT_ROOT


# ============================================================
# Locate MS-SNSD noise
# ============================================================

ms_noise_paths = sorted(
    MS_SNSD_ROOT.glob("noise_train/**/*.wav")
)

if not ms_noise_paths:
    raise RuntimeError(
        f"No MS-SNSD noise found in:\n"
        f"{MS_SNSD_ROOT / 'noise_train'}"
    )

print("\n" + "=" * 70)
print("G7 MS-SNSD + REAL-NOISE FINE-TUNE DATASET -- FULL SNR RANGE")
print("=" * 70)

print(f"\nMS-SNSD root:")
print(MS_SNSD_ROOT)

print(f"MS-SNSD noise files: {len(ms_noise_paths)}")

print(f"\nSNR range: {b.SNR_MIN} .. {b.SNR_MAX} dB  (was fixed at 0.0 dB)")


# ============================================================
# Check MS-SNSD sample rate
# ============================================================

import soundfile as sf

bad_sr = []

for p in ms_noise_paths:
    try:
        info = sf.info(str(p))
        if info.samplerate != b.FS:
            bad_sr.append(
                (str(p), info.samplerate)
            )
    except Exception as e:
        raise RuntimeError(
            f"Could not inspect {p}: {e}"
        )

if bad_sr:
    print("\nERROR: MS-SNSD contains non-16kHz files.")
    print("First few:")
    for item in bad_sr[:10]:
        print(item)
    raise RuntimeError(
        "Stopping before generating anything."
    )

print(f"MS-SNSD sample rate: {b.FS} Hz")


# ============================================================
# Build H3 speech pools
# ============================================================

print("\nLoading H3 metadata...")

h3_metadata = b.load_h3_metadata()

print("Building H3 training speech pool...")

train_speech = b.build_speech_pool(
    "train",
    h3_metadata,
)

print(
    f"H3 training speech before speaker filtering: "
    f"{len(train_speech)}"
)

print("Building H3 validation speech pool...")

val_speech = b.build_speech_pool(
    "val",
    h3_metadata,
)

print(
    f"H3 validation speech: "
    f"{len(val_speech)}"
)

# Remove the known overlapping speakers.
train_speech, removed_speakers = (
    b.remove_speaker_overlap(
        train_speech,
        val_speech,
    )
)

print(
    f"H3 training speech after speaker filtering: "
    f"{len(train_speech)}"
)

print(
    f"Removed overlapping speakers: "
    f"{removed_speakers}"
)

b.check_speech_leakage(
    train_speech,
    val_speech,
)

print("H3 speaker leakage: 0")


# ============================================================
# Build MAD + DEMAND noise
# ============================================================

print("\nBuilding MAD + DEMAND noise index...")

all_real_noise = b.build_noise_index()

train_real, val_real, test_real = (
    b.split_noise(all_real_noise)
)

print(f"Real TRAIN noise: {len(train_real)}")
print(f"Real VAL noise:   {len(val_real)}")
print(f"Real TEST noise:  {len(test_real)}")


# ============================================================
# Convert MS-SNSD into RHEAR noise-record format
# ============================================================

ms_noise = []

for p in ms_noise_paths:

    ms_noise.append(
        {
            "path": p,
            "audio": None,
            "cls": "ms_snsd",
            "source": f"ms_snsd:train:{p.stem}",
            "source_group": f"ms_snsd:train:{p.parent.name}",
            "corpus": "ms_snsd",
            "provenance": "MS-SNSD training noise",
        }
    )

print(
    f"\nMS-SNSD records prepared: "
    f"{len(ms_noise)}"
)


# ============================================================
# Make mixed training noise pool
#
# 70% MAD + DEMAND
# 30% MS-SNSD
#
# Unchanged from your original -- only the SNR range differs.
# ============================================================

rng = random.Random(SEED)

target_real_count = 7000
target_ms_count = 3000

real_selection = [
    train_real[
        rng.randrange(len(train_real))
    ]
    for _ in range(target_real_count)
]

ms_selection = [
    ms_noise[
        rng.randrange(len(ms_noise))
    ]
    for _ in range(target_ms_count)
]

combined_noise = (
    real_selection
    + ms_selection
)

rng.shuffle(combined_noise)

print("\nFine-tune noise mixture:")
print(f"  Real MAD/DEMAND: {target_real_count}")
print(f"  MS-SNSD:         {target_ms_count}")
print(f"  Total pool:      {len(combined_noise)}")


# ============================================================
# Generate 10k mixtures
# ============================================================

print(f"\nTraining SNR: {b.SNR_MIN} .. {b.SNR_MAX} dB (full range, not fixed)")
print(f"Training mixtures: {COUNT}")
print("Starting generation...\n")

result = b.generate_split(
    name="train",
    count=COUNT,
    speech_pool=train_speech,
    noise_pool=combined_noise,
    seed=SEED,
)

rows = result["rows"]

print(
    f"\nGenerated rows: {len(rows)}"
)


# ============================================================
# Validate
# ============================================================

b.validate_rows(
    rows,
    len(rows),
    "train",
)

b.validate_audio(
    rows,
    "train",
)

print("\nValidation: PASS")


# ============================================================
# Write manifest
# ============================================================

b.write_manifest(
    rows,
    OUTPUT_ROOT / "manifest.jsonl",
)


# ============================================================
# Save summary
# ============================================================

summary = {
    "dataset_name": "g7_finetune_ms_snsd_fullrange",
    "purpose": "G7 fine-tuning -- full SNR range (fix for the 0dB regression)",
    "sample_rate": b.FS,
    "duration_seconds": b.DURATION_SECONDS,
    "snr_min_db": b.SNR_MIN,
    "snr_max_db": b.SNR_MAX,
    "generated_samples": len(rows),

    "speech_source": "H3",
    "speech_pool": len(train_speech),

    "real_noise_source": "MAD + DEMAND",
    "real_noise_weight": 0.70,

    "ms_snsd_noise_source": "MS-SNSD noise_train",
    "ms_snsd_noise_files": len(ms_noise),
    "ms_snsd_weight": 0.30,

    "clip_probability": b.CLIP_PROB,
    "distortion_probability": b.DISTORT_PROB,

    "rir_enabled": False,

    "architecture_changed": False,

    "seed": SEED,

    "note": "This dataset differs from g7_finetune_ms_snsd_0db in exactly one "
            "variable: SNR range (-10..20 dB here, was fixed 0 dB). Everything "
            "else -- noise sources, mix ratio, augmentation, seed -- is identical.",
}

with (
    OUTPUT_ROOT / "dataset_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print("\n" + "=" * 70)
print("DONE.")
print(f"Output: {OUTPUT_ROOT}")
print("Next: python3 add_g7_finetune_val_fullrange.py")
print("=" * 70)
