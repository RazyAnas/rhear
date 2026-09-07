"""Validation set for the full-range fine-tune dataset.

Your add_g7_finetune_val.py, pointed at the new g7_finetune_ms_snsd_fullrange
folder instead of g7_finetune_ms_snsd_0db, with the SNR fixed to match
make_g7_finetune_fullrange.py's train set (-10..20 dB instead of a fixed 0 dB).

Everything else is identical to your original, including the one thing worth
calling out explicitly: validation uses ONLY the real MAD/DEMAND val pool, not
MS-SNSD -- same as your original design, unchanged here.

Note this dataset's manifest has TRAIN and VAL rows only, no TEST split -- same
as your original. Do not evaluate against it. Evaluate the trained checkpoint
against our standard sets instead (see the README in this folder) -- that is
what makes the result comparable to G2 / G7-base / your rejected 0dB run.

Run after make_g7_finetune_fullrange.py:
    python3 add_g7_finetune_val_fullrange.py
"""

from pathlib import Path
import sys
import json

E03_DIR = Path(__file__).resolve().parent
REPO_ROOT = E03_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

import make_g7_realnoise as b


# ============================================================
# Existing dataset — DO NOT overwrite
# ============================================================

DATASET_ROOT = (
    b.G7_DATASET
    / "PROCESSED"
    / "g7_finetune_ms_snsd_fullrange"          # matches the train-set script
)

TRAIN_JSONL = DATASET_ROOT / "train.jsonl"
TRAIN_DIR = DATASET_ROOT / "train"

if not DATASET_ROOT.exists():
    raise RuntimeError(
        f"Dataset not found:\n{DATASET_ROOT}\n"
        "Run make_g7_finetune_fullrange.py first."
    )

if not TRAIN_JSONL.exists():
    raise RuntimeError(
        f"train.jsonl not found:\n{TRAIN_JSONL}"
    )


# ============================================================
# Configuration
# ============================================================

VAL_COUNT = 1000
SEED = 46                                       # same seed as your original

# THE ONE VARIABLE -- matches make_g7_finetune_fullrange.py's train set
b.SNR_MIN = -10.0
b.SNR_MAX = 20.0

# Same augmentation as training
b.CLIP_PROB = 0.15
b.DISTORT_PROB = 0.10

b.OUTPUT_ROOT = DATASET_ROOT


# ============================================================
# Safety
# ============================================================

VAL_DIR = DATASET_ROOT / "val"
VAL_JSONL = DATASET_ROOT / "val.jsonl"

if VAL_DIR.exists() or VAL_JSONL.exists():
    raise RuntimeError(
        "Validation data already exists. "
        "Refusing to overwrite."
    )


print("\n" + "=" * 70)
print("ADDING G7 FULL-RANGE VALIDATION SET")
print("=" * 70)


# ============================================================
# H3 speech
# ============================================================

print("\nLoading H3 metadata...")

metadata = b.load_h3_metadata()

print("Building H3 training pool...")
train_speech = b.build_speech_pool(
    "train",
    metadata,
)

print("Building H3 validation pool...")
val_speech = b.build_speech_pool(
    "val",
    metadata,
)

print(f"H3 train before filtering: {len(train_speech)}")
print(f"H3 validation:              {len(val_speech)}")


# Remove speaker overlap exactly as before.

train_speech, removed = (
    b.remove_speaker_overlap(
        train_speech,
        val_speech,
    )
)

b.check_speech_leakage(
    train_speech,
    val_speech,
)

print(
    f"H3 training after filtering: {len(train_speech)}"
)

print(
    f"Removed speakers: {removed}"
)

print("Speaker overlap: 0")


# ============================================================
# MAD + DEMAND noise
# ============================================================

print("\nBuilding MAD + DEMAND noise index...")

all_noise = b.build_noise_index()

train_noise, val_noise, test_noise = (
    b.split_noise(all_noise)
)

print(f"TRAIN noise: {len(train_noise)}")
print(f"VAL noise:   {len(val_noise)}")
print(f"TEST noise:  {len(test_noise)}")


# ============================================================
# Generate validation
#
# IMPORTANT (unchanged from your original):
# Validation uses ONLY the authoritative VAL noise pool.
# It does NOT use MS-SNSD.
# ============================================================

print("\nGenerating validation mixtures...")
print(f"SNR: {b.SNR_MIN} .. {b.SNR_MAX} dB (full range, not fixed)")
print(f"Count: {VAL_COUNT}")

result = b.generate_split(
    name="val",
    count=VAL_COUNT,
    speech_pool=val_speech,
    noise_pool=val_noise,
    seed=SEED,
)

rows = result["rows"]

print(f"\nGenerated validation rows: {len(rows)}")


# ============================================================
# Validate
# ============================================================

b.validate_rows(
    rows,
    VAL_COUNT,
    "val",
)

b.validate_audio(
    rows,
    "val",
)

print("Validation audio: PASS")


# ============================================================
# Write val.jsonl
# ============================================================

b.write_manifest(
    rows,
    VAL_JSONL,
)

print(f"\nWritten:")
print(VAL_JSONL)


# ============================================================
# Rebuild combined manifest
#
# Preserve the existing 10k train rows.
# Add the 1k validation rows.
# ============================================================

print("\nRebuilding combined manifest...")

train_rows = []

with TRAIN_JSONL.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        obj = json.loads(line)

        if "_header" not in obj:
            train_rows.append(obj)


# Read validation rows.

val_rows = []

with VAL_JSONL.open(
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        obj = json.loads(line)

        if "_header" not in obj:
            val_rows.append(obj)


# ============================================================
# Construct correct header
# ============================================================

header = {
    "_header": {
        "rhear_dataset_version": "0.1.0",
        "dataset_name": "g7_finetune_ms_snsd_fullrange",
        "description": (
            "G7 fine-tuning dataset using H3 clean speech, "
            "MS-SNSD noise, MAD noise and DEMAND noise. "
            "Full -10..20 dB SNR range (fix for the 0dB regression "
            "measured against G7-base: dPESQ -0.05/-0.11, "
            "dSI-SAR -0.92/-2.27 dB on our standard eval sets)."
        ),
        "configuration": {
            "sample_rate": 16000,
            "duration_seconds": 4.0,
            "snr_range_db": [b.SNR_MIN, b.SNR_MAX],
            "noise_layers": "1-3",
            "clip_probability": 0.15,
            "distortion_probability": 0.10,
            "rir_enabled": False,
            "architecture_changed": False,
            "training_samples": len(train_rows),
            "validation_samples": len(val_rows),
        },
        "sources": {
            "H3": {
                "role": "clean speech",
            },
            "MS-SNSD": {
                "role": "training noise diversity",
            },
            "MAD": {
                "role": "real military noise",
            },
            "DEMAND": {
                "role": "real environmental noise",
            },
        },
    }
}


# ============================================================
# Write combined manifest
# ============================================================

MANIFEST = DATASET_ROOT / "manifest.jsonl"

with MANIFEST.open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        json.dumps(
            header,
            ensure_ascii=False,
        )
        + "\n"
    )

    for row in train_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

    for row in val_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Final checks
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SET COMPLETE")
print("=" * 70)

print(f"\nTRAIN: {len(train_rows)}")
print(f"VAL:   {len(val_rows)}")
print(f"TOTAL: {len(train_rows) + len(val_rows)}")

print(f"\nManifest:")
print(MANIFEST)

print("\nSNR:")
print(f"  TRAIN = {b.SNR_MIN} .. {b.SNR_MAX} dB")
print(f"  VAL   = {b.SNR_MIN} .. {b.SNR_MAX} dB")

print("\nArchitecture changed: NO")
print("Existing 0dB dataset: PRESERVED (different folder)")
print("Existing test set: PRESERVED")

print("\nREADY FOR FINE-TUNING.")
