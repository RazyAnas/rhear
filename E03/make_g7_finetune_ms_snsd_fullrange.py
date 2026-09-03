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
    / "g7_finetune_ms_snsd_fullrange"
)

COUNT = 10_000
SEED = 45

MS_SNSD_ROOT = (
    REPO_ROOT.parent
    / "MS-SNSD-master"
    / "MS-SNSD-master"
)

# Exact target condition

b.SNR_MIN = -10.0
b.SNR_MAX = 20.0

# Same augmentation as original training
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
print("G7 MS-SNSD + REAL-NOISE FINE-TUNE DATASET")
print("=" * 70)

print(f"\nMS-SNSD root:")
print(MS_SNSD_ROOT)

print(f"MS-SNSD noise files: {len(ms_noise_paths)}")


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
# We construct a repeated pool rather than copying audio.
# The mixer will randomly sample from it.
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

print("\nTraining SNR: -10 to +20 dB")
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
    "purpose": "G7 fine-tuning",
    "sample_rate": b.FS,
    "duration_seconds": b.DURATION_SECONDS,
    "snr_min_db": -10.0,
    "snr_max_db": 20.0,
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
print("FINE-TUNE DATASET COMPLETE")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_ROOT)

print(f"\nTraining mixtures: {len(rows)}")
print("SNR: -10 to +20 dB")
print("MAD/DEMAND: 70% of noise sampling")
print("MS-SNSD: 30% of noise sampling")
print("H3 speaker leakage: 0")
print("Architecture changed: NO")

print("\nREADY FOR FINE-TUNING.")