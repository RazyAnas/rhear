from pathlib import Path
import sys
import json
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------
# Import existing G7 builder + existing RHEAR mixer
# ---------------------------------------------------------------------

E03_DIR = Path(__file__).resolve().parent
REPO_ROOT = E03_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

import make_g7_realnoise as b


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEST_COUNT = 1000
TEST_SEED = 44

OUTPUT_ROOT = (
    b.G7_DATASET
    / "PROCESSED"
    / "g7_realnoise_test_0db"
)

# Force the existing mixer to generate exactly 0 dB SNR.
b.SNR_MIN = 0.0
b.SNR_MAX = 0.0

# Keep the same mixer behavior as training.
b.CLIP_PROB = 0.15
b.DISTORT_PROB = 0.10


# ---------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------

if OUTPUT_ROOT.exists():
    raise RuntimeError(
        f"\nOutput already exists:\n{OUTPUT_ROOT}\n"
        "Refusing to overwrite it."
    )

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)

# Tell the existing generate_split() to use our new directory.
b.OUTPUT_ROOT = OUTPUT_ROOT


# ---------------------------------------------------------------------
# Load H3
# ---------------------------------------------------------------------

print("\n" + "=" * 70)
print("G7 HELD-OUT REAL-NOISE TEST")
print("=" * 70)

print("\nLoading H3 metadata...")
h3_metadata = b.load_h3_metadata()

print("Building H3 validation speech pool...")
speech_pool = b.build_speech_pool(
    "val",
    h3_metadata,
)

print(f"H3 validation speech files: {len(speech_pool)}")


# ---------------------------------------------------------------------
# Build complete noise index
# ---------------------------------------------------------------------

print("\nBuilding noise index...")
noise = b.build_noise_index()

print(f"Total noise files: {len(noise)}")


# ---------------------------------------------------------------------
# Apply authoritative train/val/test noise split
# ---------------------------------------------------------------------

print("\nApplying authoritative noise split...")

train_noise, val_noise, test_noise = b.split_noise(noise)

print(f"TRAIN noise: {len(train_noise)}")
print(f"VAL noise:   {len(val_noise)}")
print(f"TEST noise:  {len(test_noise)}")


if not test_noise:
    raise RuntimeError("TEST noise pool is empty.")


# ---------------------------------------------------------------------
# Generate test mixtures
#
# H3 validation speech
# +
# TEST noise
# +
# exactly 0 dB SNR
# ---------------------------------------------------------------------

print("\nGenerating held-out test mixtures...")
print("SNR: exactly 0 dB")
print(f"Count: {TEST_COUNT}")

result = b.generate_split(
    name="test",
    count=TEST_COUNT,
    speech_pool=speech_pool,
    noise_pool=test_noise,
    seed=TEST_SEED,
)


rows = result["rows"]

print(f"\nGenerated test rows: {len(rows)}")


# ---------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------

print("\nValidating generated test dataset...")

b.validate_rows(
    rows,
    len(rows),
    "test",
)

b.validate_audio(
    rows,
    "test",
)

print("Audio validation: PASS")
print("Manifest validation: PASS")


# ---------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------

b.write_manifest(
    rows,
    OUTPUT_ROOT / "manifest.jsonl",
)


# Also save test-only JSONL if generate_split didn't already.
test_jsonl = OUTPUT_ROOT / "test.jsonl"

if not test_jsonl.exists():
    with test_jsonl.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    b.json_safe(row),
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

summary = {
    "dataset_name": "g7_realnoise_test_0db",
    "purpose": "held_out_real_noise_evaluation",
    "sample_rate": b.FS,
    "duration_seconds": b.DURATION_SECONDS,
    "snr_db": 0.0,
    "generated_samples": len(rows),
    "speech_source": "H3 validation",
    "speech_source_pool": len(speech_pool),
    "noise_source": "MAD + DEMAND",
    "noise_split": "TEST",
    "test_noise_pool": len(test_noise),
    "clip_probability": b.CLIP_PROB,
    "distortion_probability": b.DISTORT_PROB,
    "rir_enabled": False,
    "architecture_changed": False,
    "seed": TEST_SEED,
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
print("TEST DATASET COMPLETE")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_ROOT)

print(f"\nTest samples: {len(rows)}")
print("SNR: exactly 0 dB")
print("Noise: held-out TEST split")
print("Speech: H3 validation pool")
print("Architecture changed: NO")

print("\nREADY FOR EVALUATION.")