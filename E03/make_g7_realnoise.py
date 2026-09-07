#!/usr/bin/env python3
"""
RHEAR G7 REAL-NOISE DATASET BUILDER
===================================

Builds a leakage-safe G7 training/validation dataset using:

    CLEAN SPEECH
        H3 20k clean speech

    REAL NOISE
        MAD
        DEMAND

The existing RHEAR mixer is used unchanged.

IMPORTANT:
    This script does NOT modify the G7 architecture.
    It only creates the training/validation audio dataset.

Dataset layout expected:

C:\\codes\\rhear\\
├── handoff\\
│   └── data\\
│       └── h3_20k\\
│           ├── train\\
│           ├── val\\
│           └── manifest.jsonl
│
└── rhear-main\\
    ├── rhear-main\\
    │   └── E03\\
    │       ├── make_g7_realnoise.py
    │       └── rhear_data\\
    │           ├── mixing.py
    │           └── build.py
    │
    └── G7_DATASET\\
        ├── RAW\\
        │   ├── MAD\\
        │   └── DEMAND\\
        │
        ├── SPLITS\\
        │   ├── train.csv
        │   ├── validation.csv
        │   └── test.csv
        │
        └── PROCESSED\\

Generated output:

G7_DATASET\\PROCESSED\\g7_realnoise_v1\\
├── train\\
│   ├── train_000000_clean.flac
│   ├── train_000000_noisy.flac
│   └── ...
├── val\\
│   ├── val_000000_clean.flac
│   ├── val_000000_noisy.flac
│   └── ...
├── train.jsonl
├── val.jsonl
├── manifest.jsonl
└── dataset_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


# =============================================================================
# 1. PATH DETECTION
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()

# Script:
#   C:\codes\rhear\rhear-main\rhear-main\E03\make_g7_realnoise.py
#
# Therefore:
#   E03_DIR   = ...\rhear-main\rhear-main\E03
#   REPO_ROOT = ...\rhear-main\rhear-main

E03_DIR = SCRIPT_PATH.parent
REPO_ROOT = E03_DIR.parent


def find_g7_dataset() -> Path:
    """
    Find the actual G7_DATASET directory.

    Your repository is nested, so we deliberately check both levels.
    """

    candidates = [
        REPO_ROOT / "G7_DATASET",
        REPO_ROOT.parent / "G7_DATASET",
        Path(r"C:\codes\rhear\rhear-main\G7_DATASET"),
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    raise RuntimeError(
        "\nCould not find G7_DATASET.\n"
        "Expected something like:\n"
        r"C:\codes\rhear\rhear-main\G7_DATASET"
    )


def find_h3_root() -> Path:
    """
    Find the REAL H3 dataset.

    We intentionally require train/, val/ and manifest.jsonl so that
    the broken nested placeholder directory is not accidentally selected.
    """

    candidates = [
        REPO_ROOT.parent / "handoff" / "data" / "h3_20k",
        REPO_ROOT.parent.parent / "handoff" / "data" / "h3_20k",
        Path(r"C:\codes\rhear\handoff\data\h3_20k"),
    ]

    for candidate in candidates:
        if (
            candidate.is_dir()
            and (candidate / "train").is_dir()
            and (candidate / "val").is_dir()
            and (candidate / "manifest.jsonl").is_file()
        ):
            return candidate.resolve()

    # Last-resort search.
    search_root = Path(r"C:\codes\rhear")

    if search_root.is_dir():
        for candidate in search_root.glob("**/h3_20k"):
            if (
                candidate.is_dir()
                and (candidate / "train").is_dir()
                and (candidate / "val").is_dir()
                and (candidate / "manifest.jsonl").is_file()
            ):
                return candidate.resolve()

    raise RuntimeError(
        "\nCould not find the REAL H3 dataset.\n"
        "Expected:\n"
        r"C:\codes\rhear\handoff\data\h3_20k"
    )


G7_DATASET = find_g7_dataset()
H3_ROOT = find_h3_root()

MAD_ROOT = G7_DATASET / "RAW" / "MAD"
DEMAND_ROOT = G7_DATASET / "RAW" / "DEMAND"

SPLITS_ROOT = G7_DATASET / "SPLITS"

OUTPUT_ROOT = (
    G7_DATASET
    / "PROCESSED"
    / "g7_realnoise_v1"
)

H3_MANIFEST = H3_ROOT / "manifest.jsonl"

TRAIN_NOISE_CSV = SPLITS_ROOT / "train.csv"
VAL_NOISE_CSV = SPLITS_ROOT / "validation.csv"
TEST_NOISE_CSV = SPLITS_ROOT / "test.csv"


# =============================================================================
# 2. DATASET CONFIGURATION
# =============================================================================

FS = 16000

DURATION_SECONDS = 4.0

EXPECTED_SAMPLES = int(
    FS * DURATION_SECONDS
)

TRAIN_COUNT = 20_000

VAL_COUNT = 1_000

SEED = 42

SNR_MIN = -10.0
SNR_MAX = 20.0

CLIP_PROB = 0.15

DISTORT_PROB = 0.10

# IMPORTANT:
#
# DEMAND files are roughly 5 minutes long.
# One 5-minute mono float32 recording is roughly:
#
#   300 * 16000 * 4 ≈ 19.2 MB
#
# 128 recordings can therefore consume several GB.
#
# 16 is much safer.

CACHE_SIZE = 16


# =============================================================================
# 3. IMPORT EXISTING RHEAR DATA PIPELINE
# =============================================================================

sys.path.insert(0, str(REPO_ROOT))

try:
    from E03.rhear_data import build as rhear_build
    from E03.rhear_data import mixing

except Exception as exc:
    raise RuntimeError(
        "\nCould not import the existing RHEAR data modules.\n"
        f"Repository root:\n{REPO_ROOT}\n\n"
        "Expected:\n"
        "E03/rhear_data/build.py\n"
        "E03/rhear_data/mixing.py"
    ) from exc


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def norm_path(path: Path | str) -> str:
    """
    Normalize paths so Windows path comparisons are reliable.
    """

    return (
        str(Path(path).resolve())
        .replace("/", "\\")
        .lower()
    )


def json_safe(value: Any) -> Any:
    """
    Convert NumPy and Path objects into JSON-safe Python objects.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(
            json_safe(item)
            for item in value
        )

    return value


def find_audio(root: Path) -> list[Path]:
    """
    Recursively find WAV/FLAC files.
    """

    if not root.is_dir():
        raise FileNotFoundError(
            f"Directory does not exist:\n{root}"
        )

    files = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in {".wav", ".flac"}
        )
    ]

    return sorted(files)


# =============================================================================
# 5. AUDIO LOADING
# =============================================================================

def load_audio(path: Path) -> np.ndarray:
    """
    Load audio as mono float32 at 16 kHz.

    Stereo -> mono
    Other sample rates -> resampled to 16 kHz
    """

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    # Stereo/multichannel -> mono.
    if audio.ndim == 2:
        audio = np.mean(
            audio,
            axis=1,
            dtype=np.float32,
        )

    if audio.ndim != 1:
        raise RuntimeError(
            f"Unexpected audio shape "
            f"{audio.shape}:\n{path}"
        )

    if not np.isfinite(audio).all():
        raise RuntimeError(
            f"Non-finite audio detected:\n{path}"
        )

    # Resample only when required.
    if sample_rate != FS:

        from scipy.signal import resample_poly

        gcd = math.gcd(
            int(sample_rate),
            FS,
        )

        audio = resample_poly(
            audio,
            FS // gcd,
            int(sample_rate) // gcd,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    return np.ascontiguousarray(
        audio,
        dtype=np.float32,
    )


# =============================================================================
# 6. H3 MANIFEST
# =============================================================================

def load_h3_metadata() -> dict[str, dict[str, Any]]:
    """
    Load H3 manifest.

    Returns:
        filename -> metadata
    """

    if not H3_MANIFEST.is_file():
        raise FileNotFoundError(
            f"H3 manifest missing:\n{H3_MANIFEST}"
        )

    metadata: dict[str, dict[str, Any]] = {}

    with H3_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in H3 manifest "
                    f"at line {line_number}."
                ) from exc

            # Header line.
            if "_header" in row:
                continue

            clean = row.get("clean")

            if not clean:
                continue

            filename = Path(clean).name

            if filename in metadata:
                raise RuntimeError(
                    f"Duplicate H3 filename:\n{filename}"
                )

            metadata[filename] = row

    if not metadata:
        raise RuntimeError(
            "H3 manifest contains no usable utterances."
        )

    return metadata


# =============================================================================
# 7. BUILD H3 SPEECH POOL
# =============================================================================

def build_speech_pool(
    split: str,
    metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:

    root = H3_ROOT / split

    files = sorted(
        root.glob("*_clean.flac")
    )

    if not files:
        raise RuntimeError(
            f"No H3 clean files found:\n{root}"
        )

    pool: list[dict[str, Any]] = []

    for path in files:

        row = metadata.get(path.name)

        if row is None:
            raise RuntimeError(
                f"No H3 manifest metadata for:\n{path}"
            )

        speaker = row.get("speaker")
        utt = row.get("utt")

        if speaker is None:
            raise RuntimeError(
                f"Missing speaker metadata:\n{path}"
            )

        if utt is None:
            raise RuntimeError(
                f"Missing utterance metadata:\n{path}"
            )

        pool.append(
            {
                "path": path,
                "speaker": str(speaker),
                "utt": str(utt),
                "speech_source_path": str(
                    row.get(
                        "speech_source_path",
                        path,
                    )
                ),
            }
        )

    return pool


# =============================================================================
# 8. FIX H3 SPEAKER LEAKAGE
# =============================================================================

def remove_speaker_overlap(
    train_pool: list[dict[str, Any]],
    val_pool: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    set[str],
]:
    """
    Remove every H3 TRAIN utterance whose speaker appears in VALIDATION.

    Validation is treated as authoritative.

    This fixes the observed 5-speaker overlap.
    """

    validation_speakers = {
        str(item["speaker"])
        for item in val_pool
    }

    filtered_train = [
        item
        for item in train_pool
        if str(item["speaker"])
        not in validation_speakers
    ]

    removed_speakers = {
        str(item["speaker"])
        for item in train_pool
        if str(item["speaker"])
        in validation_speakers
    }

    return (
        filtered_train,
        removed_speakers,
    )


def check_speech_leakage(
    train_pool: list[dict[str, Any]],
    val_pool: list[dict[str, Any]],
) -> None:
    """
    Fail if train and validation share speakers.
    """

    train_speakers = {
        str(item["speaker"])
        for item in train_pool
    }

    val_speakers = {
        str(item["speaker"])
        for item in val_pool
    }

    overlap = (
        train_speakers
        & val_speakers
    )

    print(
        f"TRAIN speakers: {len(train_speakers)}"
    )

    print(
        f"VAL speakers:   {len(val_speakers)}"
    )

    print(
        f"Speaker overlap: {len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "\nH3 SPEAKER LEAKAGE DETECTED:\n"
            + "\n".join(
                sorted(overlap)
            )
        )


# =============================================================================
# 9. READ NOISE SPLIT CSV
# =============================================================================

def read_split_csv(
    csv_path: Path,
) -> set[str]:
    """
    Read a noise split CSV.

    Automatically detects a reasonable path/file column.
    """

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Noise split CSV missing:\n{csv_path}"
        )

    paths: set[str] = set()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        fields = (
            reader.fieldnames
            or []
        )

        if not fields:
            raise RuntimeError(
                f"CSV has no header:\n{csv_path}"
            )

        path_field = None

        preferred_fields = [
            "path",
            "file",
            "filepath",
            "file_path",
            "audio",
            "audio_path",
        ]

        for field in preferred_fields:
            if field in fields:
                path_field = field
                break

        # Fallback.
        if path_field is None:

            for field in fields:

                low = field.lower()

                if (
                    "path" in low
                    or "file" in low
                ):
                    path_field = field
                    break

        if path_field is None:
            raise RuntimeError(
                f"Could not identify path column in:\n"
                f"{csv_path}\n"
                f"Columns: {fields}"
            )

        for row in reader:

            value = row.get(
                path_field
            )

            if not value:
                continue

            path = Path(
                value.strip()
            )

            if not path.is_absolute():
                path = (
                    G7_DATASET / path
                )

            paths.add(
                norm_path(path)
            )

    if not paths:
        raise RuntimeError(
            f"No paths found in:\n{csv_path}"
        )

    return paths


# =============================================================================
# 10. MAD INDEX
# =============================================================================

def index_mad() -> list[dict[str, Any]]:
    """
    Index MAD.

    IMPORTANT:
    The generic RHEAR MAD index historically produced source identifiers
    such as:

        mad_clip:0

    Those are not safe for group leakage detection.

    We therefore derive the group from the real directory:

        MAD/training/022/file.wav

    ->

        mad:training:022
    """

    indexed, skipped = (
        rhear_build.index_mad(
            str(MAD_ROOT)
        )
    )

    print(
        f"MAD files skipped by indexer: "
        f"{skipped}"
    )

    result: list[dict[str, Any]] = []

    mad_root = MAD_ROOT.resolve()

    for item in indexed:

        path = Path(
            item["path"]
        ).resolve()

        try:

            relative = path.relative_to(
                mad_root
            )

        except ValueError as exc:

            raise RuntimeError(
                f"MAD file outside MAD_ROOT:\n"
                f"{path}"
            ) from exc

        # Expected:
        #
        # training / 022 / file.wav
        # test     / 052 / file.wav

        if len(relative.parts) < 3:
            raise RuntimeError(
                f"Unexpected MAD path structure:\n"
                f"{path}"
            )

        split_name = (
            relative.parts[0]
        )

        group_name = (
            relative.parts[1]
        )

        source_group = (
            f"mad:{split_name}:{group_name}"
        )

        result.append(
            {
                "path": path,
                "audio": None,
                "cls": str(
                    item.get(
                        "cls",
                        "unknown",
                    )
                ),
                "source": source_group,
                "source_group": source_group,
                "corpus": "mad",
                "provenance": str(
                    item.get(
                        "provenance",
                        "real recording",
                    )
                ),
            }
        )

    return result


# =============================================================================
# 11. DEMAND INDEX
# =============================================================================

def demand_group(
    path: Path,
) -> str:
    """
    Determine DEMAND environment.

    Expected structure is approximately:

        DEMAND/
            DKITCHEN/
                DKITCHEN/
                    *.wav

    Therefore the first directory below DEMAND is used.
    """

    try:

        relative = path.resolve().relative_to(
            DEMAND_ROOT.resolve()
        )

    except ValueError:
        return "unknown"

    if not relative.parts:
        return "unknown"

    return relative.parts[0].lower()


def index_demand() -> list[dict[str, Any]]:
    """
    Index DEMAND recordings.
    """

    files = find_audio(
        DEMAND_ROOT
    )

    if not files:
        raise RuntimeError(
            f"No DEMAND audio found:\n"
            f"{DEMAND_ROOT}"
        )

    result: list[dict[str, Any]] = []

    for path in files:

        group = demand_group(
            path
        )

        if group == "unknown":
            raise RuntimeError(
                f"Could not determine DEMAND "
                f"source group:\n{path}"
            )

        result.append(
            {
                "path": path,
                "audio": None,
                "cls": "ambient",
                "source": (
                    f"demand:{group}"
                ),
                "source_group": group,
                "corpus": "demand",
                "provenance": "real recording",
            }
        )

    return result


# =============================================================================
# 12. BUILD COMPLETE NOISE INDEX
# =============================================================================

def build_noise_index() -> list[dict[str, Any]]:

    print()
    print("Indexing MAD...")

    mad = index_mad()

    print(
        f"MAD files: {len(mad)}"
    )

    print()
    print("Indexing DEMAND...")

    demand = index_demand()

    print(
        f"DEMAND files: {len(demand)}"
    )

    noise = mad + demand

    if not noise:
        raise RuntimeError(
            "No noise files found."
        )

    return noise


# =============================================================================
# 13. APPLY PRE-CREATED NOISE SPLITS
# =============================================================================

def split_noise(
    noise: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Apply the existing leakage-safe CSV split.

    The CSVs are authoritative.
    """

    train_set = read_split_csv(
        TRAIN_NOISE_CSV
    )

    val_set = read_split_csv(
        VAL_NOISE_CSV
    )

    test_set = read_split_csv(
        TEST_NOISE_CSV
    )

    # -------------------------------------------------------------------------
    # Check CSV-to-CSV leakage.
    # -------------------------------------------------------------------------

    if train_set & val_set:
        raise RuntimeError(
            "TRAIN/VALIDATION noise file leakage "
            "exists in CSVs."
        )

    if train_set & test_set:
        raise RuntimeError(
            "TRAIN/TEST noise file leakage "
            "exists in CSVs."
        )

    if val_set & test_set:
        raise RuntimeError(
            "VALIDATION/TEST noise file leakage "
            "exists in CSVs."
        )

    # -------------------------------------------------------------------------
    # Match indexed files to CSVs.
    # -------------------------------------------------------------------------

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    missing_from_csv: list[dict[str, Any]] = []

    for item in noise:

        key = norm_path(
            item["path"]
        )

        if key in train_set:
            train.append(item)

        elif key in val_set:
            val.append(item)

        elif key in test_set:
            test.append(item)

        else:
            missing_from_csv.append(
                item
            )

    if missing_from_csv:

        print()
        print(
            "ERROR: some indexed files are not in "
            "any split CSV."
        )

        for item in missing_from_csv[:10]:
            print(
                "  ",
                item["path"],
            )

        raise RuntimeError(
            f"{len(missing_from_csv)} indexed noise files "
            "are not covered by the split CSVs."
        )

    # -------------------------------------------------------------------------
    # Check reverse direction:
    # CSV files must actually exist in indexed data.
    # -------------------------------------------------------------------------

    indexed_paths = {
        norm_path(
            item["path"]
        )
        for item in noise
    }

    for name, split_set in [
        ("TRAIN", train_set),
        ("VALIDATION", val_set),
        ("TEST", test_set),
    ]:

        nonexistent = (
            split_set
            - indexed_paths
        )

        if nonexistent:

            examples = sorted(
                nonexistent
            )[:10]

            raise RuntimeError(
                f"{name} CSV contains paths that "
                f"were not indexed.\n"
                + "\n".join(examples)
            )

    if not train:
        raise RuntimeError(
            "TRAIN noise pool is empty."
        )

    if not val:
        raise RuntimeError(
            "VALIDATION noise pool is empty."
        )

    if not test:
        raise RuntimeError(
            "TEST noise pool is empty."
        )

    return (
        train,
        val,
        test,
    )


# =============================================================================
# 14. NOISE GROUP ID
# =============================================================================

def noise_group_key(
    item: dict[str, Any],
) -> str:
    """
    Return stable group identity.
    """

    corpus = str(
        item["corpus"]
    ).lower()

    if corpus == "mad":
        return str(
            item["source"]
        )

    if corpus == "demand":
        return (
            f"demand:"
            f"{item.get('source_group', 'unknown')}"
        )

    return str(
        item.get(
            "source_group",
            item.get(
                "source",
                "unknown",
            ),
        )
    )


# =============================================================================
# 15. NOISE LEAKAGE CHECK
# =============================================================================

def check_noise_leakage(
    train_noise: list[dict[str, Any]],
    val_noise: list[dict[str, Any]],
) -> None:
    """
    Check:

        1. exact file leakage
        2. source-group leakage
    """

    train_files = {
        norm_path(
            item["path"]
        )
        for item in train_noise
    }

    val_files = {
        norm_path(
            item["path"]
        )
        for item in val_noise
    }

    file_overlap = (
        train_files
        & val_files
    )

    print(
        f"Noise file overlap: "
        f"{len(file_overlap)}"
    )

    if file_overlap:
        raise RuntimeError(
            "NOISE FILE LEAKAGE DETECTED."
        )

    train_groups = {
        noise_group_key(item)
        for item in train_noise
    }

    val_groups = {
        noise_group_key(item)
        for item in val_noise
    }

    group_overlap = (
        train_groups
        & val_groups
    )

    print(
        f"Noise source-group overlap: "
        f"{len(group_overlap)}"
    )

    if group_overlap:
        raise RuntimeError(
            "NOISE SOURCE-GROUP LEAKAGE DETECTED:\n"
            + "\n".join(
                sorted(group_overlap)
            )
        )


# =============================================================================
# 16. MEMORY-SAFE AUDIO CACHE
# =============================================================================

class AudioCache:
    """
    Small FIFO cache.

    Deliberately kept small because DEMAND recordings are long.
    """

    def __init__(
        self,
        max_items: int = CACHE_SIZE,
    ):

        if max_items < 1:
            raise ValueError(
                "max_items must be >= 1"
            )

        self.max_items = max_items

        self.data: dict[
            str,
            np.ndarray,
        ] = {}

    def get(
        self,
        path: Path,
    ) -> np.ndarray:

        key = norm_path(path)

        if key in self.data:
            return self.data[key]

        audio = load_audio(path)

        # FIFO eviction.
        if len(self.data) >= self.max_items:

            oldest = next(
                iter(self.data)
            )

            del self.data[oldest]

        self.data[key] = audio

        return audio


# =============================================================================
# 17. MIX ONE EXAMPLE
# =============================================================================

def make_one_mix(
    speech_entry: dict[str, Any],
    noise_pool: list[dict[str, Any]],
    rng: np.random.Generator,
    cache: AudioCache,
):
    """
    Generate one clean/noisy training example.

    Uses the EXISTING RHEAR mixer.

    No architecture changes occur here.
    """

    speech_audio = cache.get(
        speech_entry["path"]
    )

    speech = {
        "audio": speech_audio,
        "speaker": speech_entry[
            "speaker"
        ],
        "utt": speech_entry[
            "utt"
        ],
        "source": "H3",
        "path": str(
            speech_entry["path"]
        ),
        "provenance": "real recording",
    }

    # Existing mixer supports 1-3 noise recordings.
    max_layers = min(
        3,
        len(noise_pool),
    )

    number_of_layers = int(
        rng.integers(
            1,
            max_layers + 1,
        )
    )

    selected_indices = rng.choice(
        len(noise_pool),
        size=number_of_layers,
        replace=False,
    )

    noises = []

    selected_items = []

    for index in selected_indices:

        item = noise_pool[
            int(index)
        ]

        audio = cache.get(
            item["path"]
        )

        noises.append(
            {
                "audio": audio,
                "cls": item["cls"],
                "source": item["source"],
                "provenance": item.get(
                    "provenance",
                    "real recording",
                ),
            }
        )

        selected_items.append(
            item
        )

    # IMPORTANT:
    #
    # Existing RHEAR mixer API:
    #
    #     noisy, target, metadata
    #
    # We preserve this exactly.

    noisy, clean, metadata = (
        mixing.make_mixture(
            speech=speech,
            noises=noises,
            fs=FS,
            rng=rng,
            rir=None,
            dur_s=DURATION_SECONDS,
            snr_range=(
                SNR_MIN,
                SNR_MAX,
            ),
            clip_prob=CLIP_PROB,
            distort_prob=DISTORT_PROB,
        )
    )

    clean = np.asarray(
        clean,
        dtype=np.float32,
    )

    noisy = np.asarray(
        noisy,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------------
    # Shape validation.
    # -------------------------------------------------------------------------

    if clean.ndim != 1:
        raise RuntimeError(
            f"Mixer returned invalid clean shape: "
            f"{clean.shape}"
        )

    if noisy.ndim != 1:
        raise RuntimeError(
            f"Mixer returned invalid noisy shape: "
            f"{noisy.shape}"
        )

    if len(clean) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Clean sample count is "
            f"{len(clean)}, expected "
            f"{EXPECTED_SAMPLES}"
        )

    if len(noisy) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Noisy sample count is "
            f"{len(noisy)}, expected "
            f"{EXPECTED_SAMPLES}"
        )

    # -------------------------------------------------------------------------
    # Numerical validation.
    # -------------------------------------------------------------------------

    if not np.isfinite(
        clean
    ).all():

        raise RuntimeError(
            "Mixer produced non-finite CLEAN audio."
        )

    if not np.isfinite(
        noisy
    ).all():

        raise RuntimeError(
            "Mixer produced non-finite NOISY audio."
        )

    return (
        clean,
        noisy,
        metadata,
        selected_items,
    )


# =============================================================================
# 18. MANIFEST HEADER
# =============================================================================

def manifest_header() -> dict[str, Any]:
    """
    Correct dataset provenance information.

    No stale LibriSpeech/MUSAN/RIRS claims.
    """

    return {
        "rhear_dataset_version": "0.1.0",
        "dataset_name": "g7_realnoise_v1",

        "description": (
            "Leakage-safe synthetic noisy speech dataset "
            "for RHEAR G7, generated by mixing H3 clean "
            "speech with real MAD and DEMAND recordings."
        ),

        "configuration": {
            "sample_rate": FS,
            "duration_seconds": DURATION_SECONDS,
            "snr_range_db": [
                SNR_MIN,
                SNR_MAX,
            ],
            "noise_layers": "1-3",
            "clip_probability": CLIP_PROB,
            "distortion_probability": DISTORT_PROB,
            "rir_enabled": False,
            "random_seed_train": SEED,
            "random_seed_validation": SEED + 1,
            "architecture_changed": False,
        },

        "sources": {

            "H3": {
                "role": "clean speech",
                "description": (
                    "H3 clean speech dataset"
                ),
            },

            "MAD": {
                "role": "real noise",
                "description": (
                    "Military Audio Dataset"
                ),
                "licence": "CC BY 4.0",
            },

            "DEMAND": {
                "role": "real environmental noise",
                "description": (
                    "Diverse Environments Multi-channel "
                    "Acoustic Noise Database"
                ),
                "licence": "CC BY-SA 3.0",
            },
        },
    }


# =============================================================================
# 19. WRITE MANIFEST
# =============================================================================

def write_manifest(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """
    Write:

        _header
        row
        row
        row
        ...

    JSONL format.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:

        header = {
            "_header": json_safe(
                manifest_header()
            )
        }

        file.write(
            json.dumps(
                header,
                ensure_ascii=False,
            )
            + "\n"
        )

        for row in rows:

            safe_row = json_safe(
                row
            )

            file.write(
                json.dumps(
                    safe_row,
                    ensure_ascii=False,
                )
                + "\n"
            )


# =============================================================================
# 20. OUTPUT SAFETY
# =============================================================================

def check_output_is_clean() -> None:
    """
    Refuse to reuse a partially generated dataset.

    This prevents old files from being silently mixed with new files.
    """

    if not OUTPUT_ROOT.exists():
        return

    generated_files = [
        path
        for path in OUTPUT_ROOT.rglob("*")
        if (
            path.is_file()
            and (
                path.name.endswith(
                    "_clean.flac"
                )
                or path.name.endswith(
                    "_noisy.flac"
                )
                or path.name in {
                    "manifest.jsonl",
                    "train.jsonl",
                    "val.jsonl",
                    "dataset_summary.json",
                }
            )
        )
    ]

    if generated_files:

        raise RuntimeError(
            "\nOutput directory already contains "
            "generated files:\n\n"
            f"{OUTPUT_ROOT}\n\n"
            "Delete it before running again:\n\n"
            f'Remove-Item -Recurse -Force "{OUTPUT_ROOT}"'
        )


# =============================================================================
# 21. GENERATE SPLIT
# =============================================================================

def generate_split(
    name: str,
    count: int,
    speech_pool: list[dict[str, Any]],
    noise_pool: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """
    Generate one complete split.
    """

    if count < 1:
        raise ValueError(
            "count must be >= 1"
        )

    if not speech_pool:
        raise RuntimeError(
            f"{name}: speech pool is empty."
        )

    if not noise_pool:
        raise RuntimeError(
            f"{name}: noise pool is empty."
        )

    print()
    print("=" * 70)
    print(
        f"GENERATING {name.upper()}"
    )
    print("=" * 70)

    output_dir = (
        OUTPUT_ROOT / name
    )

    # This should not exist because the top-level output is created
    # only immediately before generation.
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rng = np.random.default_rng(
        seed
    )

    cache = AudioCache(
        CACHE_SIZE
    )

    rows: list[
        dict[str, Any]
    ] = []

    speaker_counter = Counter()

    corpus_counter = Counter()

    class_counter = Counter()

    for i in range(count):

        if (
            i % 100 == 0
            or i == count - 1
        ):

            print(
                f"\rGenerated "
                f"{i + 1:,}/{count:,}",
                end="",
                flush=True,
            )

        # ---------------------------------------------------------------
        # Random H3 source utterance.
        #
        # Sampling WITH replacement is intentional.
        # We need 20k generated mixtures even after removing the
        # overlapping speakers.
        # ---------------------------------------------------------------

        speech_entry = speech_pool[
            int(
                rng.integers(
                    0,
                    len(speech_pool),
                )
            )
        ]

        (
            clean,
            noisy,
            metadata,
            selected_noise,
        ) = make_one_mix(
            speech_entry,
            noise_pool,
            rng,
            cache,
        )

        sample_id = (
            f"{name}_{i:06d}"
        )

        clean_filename = (
            f"{sample_id}_clean.flac"
        )

        noisy_filename = (
            f"{sample_id}_noisy.flac"
        )

        clean_path = (
            output_dir
            / clean_filename
        )

        noisy_path = (
            output_dir
            / noisy_filename
        )

        # ---------------------------------------------------------------
        # Write FLAC PCM16.
        # ---------------------------------------------------------------

        sf.write(
            str(clean_path),
            clean,
            FS,
            format="FLAC",
            subtype="PCM_16",
        )

        sf.write(
            str(noisy_path),
            noisy,
            FS,
            format="FLAC",
            subtype="PCM_16",
        )

        # ---------------------------------------------------------------
        # Metadata.
        # ---------------------------------------------------------------

        row = {}

        if isinstance(
            metadata,
            dict,
        ):
            row.update(
                metadata
            )

        row.update(
            {
                "id": sample_id,

                "split": name,

                "clean": (
                    f"{name}/"
                    f"{clean_filename}"
                ),

                "noisy": (
                    f"{name}/"
                    f"{noisy_filename}"
                ),

                "speaker": str(
                    speech_entry[
                        "speaker"
                    ]
                ),

                "utt": str(
                    speech_entry[
                        "utt"
                    ]
                ),

                "speech_source": "H3",

                "speech_source_path": str(
                    speech_entry[
                        "speech_source_path"
                    ]
                ),

                "noise_corpora": sorted(
                    {
                        str(
                            item[
                                "corpus"
                            ]
                        )
                        for item in selected_noise
                    }
                ),

                "noise_source_paths": [
                    str(
                        item["path"]
                    )
                    for item in selected_noise
                ],

                "noise_classes": [
                    str(
                        item["cls"]
                    )
                    for item in selected_noise
                ],

                "noise_source_groups": [
                    noise_group_key(
                        item
                    )
                    for item in selected_noise
                ],

                "provenance": (
                    "synthetically mixed "
                    "from real recordings"
                ),
            }
        )

        row = json_safe(row)

        rows.append(row)

        speaker_counter[
            str(
                speech_entry[
                    "speaker"
                ]
            )
        ] += 1

        for item in selected_noise:

            corpus_counter[
                str(
                    item["corpus"]
                )
            ] += 1

            class_counter[
                str(
                    item["cls"]
                )
            ] += 1

    print()

    print(
        f"{name}: "
        f"{len(rows):,} samples"
    )

    # Write split manifest.
    write_manifest(
        rows,
        OUTPUT_ROOT
        / f"{name}.jsonl",
    )

    return {
        "rows": rows,
        "speakers": dict(
            speaker_counter
        ),
        "corpora": dict(
            corpus_counter
        ),
        "classes": dict(
            class_counter
        ),
    }


# =============================================================================
# 22. VALIDATE GENERATED FILES
# =============================================================================

def validate_audio(
    rows: list[dict[str, Any]],
    split: str,
) -> None:
    """
    Read representative files back from disk.

    Every generated in-memory sample is already shape/finite checked.
    This additional test confirms that the actual FLAC files written to
    disk can be read correctly.
    """

    print(
        f"Validating {split} audio..."
    )

    if not rows:
        raise RuntimeError(
            f"{split} contains no rows."
        )

    indices = sorted(
        {
            0,
            len(rows) // 2,
            len(rows) - 1,
        }
    )

    for index in indices:

        row = rows[index]

        clean_path = (
            OUTPUT_ROOT
            / row["clean"]
        )

        noisy_path = (
            OUTPUT_ROOT
            / row["noisy"]
        )

        if not clean_path.is_file():
            raise RuntimeError(
                f"Missing clean file:\n"
                f"{clean_path}"
            )

        if not noisy_path.is_file():
            raise RuntimeError(
                f"Missing noisy file:\n"
                f"{noisy_path}"
            )

        clean_audio, clean_fs = (
            sf.read(
                str(clean_path),
                dtype="float32",
                always_2d=False,
            )
        )

        noisy_audio, noisy_fs = (
            sf.read(
                str(noisy_path),
                dtype="float32",
                always_2d=False,
            )
        )

        if clean_fs != FS:
            raise RuntimeError(
                f"Wrong clean sample rate: "
                f"{clean_fs}"
            )

        if noisy_fs != FS:
            raise RuntimeError(
                f"Wrong noisy sample rate: "
                f"{noisy_fs}"
            )

        if len(clean_audio) != EXPECTED_SAMPLES:
            raise RuntimeError(
                f"Wrong clean length: "
                f"{len(clean_audio)}"
            )

        if len(noisy_audio) != EXPECTED_SAMPLES:
            raise RuntimeError(
                f"Wrong noisy length: "
                f"{len(noisy_audio)}"
            )

        if not np.isfinite(
            clean_audio
        ).all():

            raise RuntimeError(
                f"Non-finite clean audio:\n"
                f"{clean_path}"
            )

        if not np.isfinite(
            noisy_audio
        ).all():

            raise RuntimeError(
                f"Non-finite noisy audio:\n"
                f"{noisy_path}"
            )

    print(
        f"{split}: OK"
    )


# =============================================================================
# 23. VALIDATE MANIFEST ROWS
# =============================================================================

def validate_rows(
    rows: list[dict[str, Any]],
    expected_count: int,
    split: str,
) -> None:
    """
    Validate generated manifest structure.
    """

    if len(rows) != expected_count:
        raise RuntimeError(
            f"{split} generated "
            f"{len(rows)} rows; expected "
            f"{expected_count}."
        )

    ids: set[str] = set()

    for row in rows:

        row_id = str(
            row["id"]
        )

        if row_id in ids:
            raise RuntimeError(
                f"Duplicate sample ID: "
                f"{row_id}"
            )

        ids.add(row_id)

        if row["split"] != split:
            raise RuntimeError(
                f"Incorrect split for "
                f"{row_id}: "
                f"{row['split']}"
            )

        clean = str(
            row["clean"]
        )

        noisy = str(
            row["noisy"]
        )

        if not clean.startswith(
            f"{split}/"
        ):
            raise RuntimeError(
                f"Invalid clean path "
                f"for {row_id}: "
                f"{clean}"
            )

        if not noisy.startswith(
            f"{split}/"
        ):
            raise RuntimeError(
                f"Invalid noisy path "
                f"for {row_id}: "
                f"{noisy}"
            )


# =============================================================================
# 24. SUMMARY
# =============================================================================

def write_summary(
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    train_speech_pool: list[dict[str, Any]],
    val_speech_pool: list[dict[str, Any]],
    removed_speakers: set[str],
    train_noise: list[dict[str, Any]],
    val_noise: list[dict[str, Any]],
    test_noise: list[dict[str, Any]],
) -> None:
    """
    Write an accurate dataset summary.
    """

    summary = {

        "dataset": "g7_realnoise_v1",

        "architecture_changed": False,

        "sample_rate": FS,

        "duration_seconds": (
            DURATION_SECONDS
        ),

        "snr_range_db": [
            SNR_MIN,
            SNR_MAX,
        ],

        "generated_samples": {

            "train": len(
                train_result["rows"]
            ),

            "validation": len(
                val_result["rows"]
            ),

            "total": (
                len(
                    train_result["rows"]
                )
                +
                len(
                    val_result["rows"]
                )
            ),
        },

        "speech": {

            "train_source_pool_after_filter": (
                len(train_speech_pool)
            ),

            "validation_source_pool": (
                len(val_speech_pool)
            ),

            "removed_overlapping_speakers": (
                sorted(
                    removed_speakers
                )
            ),

            "speaker_disjoint": True,

            "sampling_with_replacement": True,
        },

        "noise_pool": {

            "train": len(
                train_noise
            ),

            "validation": len(
                val_noise
            ),

            "test": len(
                test_noise
            ),
        },

        "train_noise_corpora": (
            train_result["corpora"]
        ),

        "validation_noise_corpora": (
            val_result["corpora"]
        ),

        "train_noise_classes": (
            train_result["classes"]
        ),

        "validation_noise_classes": (
            val_result["classes"]
        ),

        "output": str(
            OUTPUT_ROOT
        ),
    }

    summary_path = (
        OUTPUT_ROOT
        / "dataset_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(summary),
            file,
            indent=2,
            ensure_ascii=False,
        )


# =============================================================================
# 25. MAIN
# =============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "RHEAR G7 REAL-NOISE DATASET BUILDER"
    )
    print("=" * 70)

    # =========================================================================
    # PATHS
    # =========================================================================

    print()
    print("Detected paths:")

    print(
        f"  Repository : {REPO_ROOT}"
    )

    print(
        f"  G7_DATASET : {G7_DATASET}"
    )

    print(
        f"  H3         : {H3_ROOT}"
    )

    print(
        f"  MAD        : {MAD_ROOT}"
    )

    print(
        f"  DEMAND     : {DEMAND_ROOT}"
    )

    print(
        f"  Output     : {OUTPUT_ROOT}"
    )

    # =========================================================================
    # PREFLIGHT
    # =========================================================================

    print()
    print("=" * 70)
    print("PRE-FLIGHT")
    print("=" * 70)

    required = {

        "H3": H3_ROOT,

        "H3 manifest": H3_MANIFEST,

        "H3 train": (
            H3_ROOT / "train"
        ),

        "H3 val": (
            H3_ROOT / "val"
        ),

        "MAD": MAD_ROOT,

        "DEMAND": DEMAND_ROOT,

        "TRAIN noise split": (
            TRAIN_NOISE_CSV
        ),

        "VALIDATION noise split": (
            VAL_NOISE_CSV
        ),

        "TEST noise split": (
            TEST_NOISE_CSV
        ),

        "RHEAR mixer": (
            REPO_ROOT
            / "E03"
            / "rhear_data"
            / "mixing.py"
        ),

        "RHEAR build": (
            REPO_ROOT
            / "E03"
            / "rhear_data"
            / "build.py"
        ),
    }

    for name, path in required.items():

        if not path.exists():

            raise RuntimeError(
                f"{name} is missing:\n"
                f"{path}"
            )

        print(
            f"[OK] {name}"
        )

    # =========================================================================
    # OUTPUT SAFETY
    # =========================================================================

    check_output_is_clean()

    print(
        "[OK] output directory is clean"
    )

    # =========================================================================
    # H3
    # =========================================================================

    print()
    print("=" * 70)
    print("H3 SPEECH")
    print("=" * 70)

    h3_metadata = (
        load_h3_metadata()
    )

    raw_train_speech = (
        build_speech_pool(
            "train",
            h3_metadata,
        )
    )

    val_speech = (
        build_speech_pool(
            "val",
            h3_metadata,
        )
    )

    print(
        "H3 train speech before "
        f"filtering: "
        f"{len(raw_train_speech):,}"
    )

    print(
        f"H3 validation speech: "
        f"{len(val_speech):,}"
    )

    # -------------------------------------------------------------------------
    # Remove the known 5 overlapping speakers.
    # -------------------------------------------------------------------------

    (
        train_speech,
        removed_speakers,
    ) = remove_speaker_overlap(
        raw_train_speech,
        val_speech,
    )

    print(
        "Overlapping speakers removed "
        f"from H3 train: "
        f"{len(removed_speakers)}"
    )

    if removed_speakers:

        print(
            "Removed speaker IDs:"
        )

        for speaker in sorted(
            removed_speakers
        ):

            print(
                f"  {speaker}"
            )

    print(
        "H3 train speech after "
        f"filtering: "
        f"{len(train_speech):,}"
    )

    if not train_speech:
        raise RuntimeError(
            "No H3 training speech remains "
            "after speaker filtering."
        )

    # -------------------------------------------------------------------------
    # HARD SPEAKER LEAKAGE CHECK.
    # -------------------------------------------------------------------------

    print()
    print(
        "Checking H3 speaker leakage..."
    )

    check_speech_leakage(
        train_speech,
        val_speech,
    )

    # =========================================================================
    # NOISE
    # =========================================================================

    print()
    print("=" * 70)
    print("REAL NOISE")
    print("=" * 70)

    noise = build_noise_index()

    print()
    print(
        f"Total indexed noise: "
        f"{len(noise):,}"
    )

    (
        train_noise,
        val_noise,
        test_noise,
    ) = split_noise(
        noise
    )

    print(
        f"TRAIN noise: "
        f"{len(train_noise):,}"
    )

    print(
        f"VAL noise:   "
        f"{len(val_noise):,}"
    )

    print(
        f"TEST noise:  "
        f"{len(test_noise):,}"
    )

    # =========================================================================
    # NOISE LEAKAGE CHECK
    # =========================================================================

    print()
    print(
        "Checking noise leakage..."
    )

    check_noise_leakage(
        train_noise,
        val_noise,
    )

    # =========================================================================
    # FINAL PREFLIGHT
    # =========================================================================

    print()
    print("=" * 70)
    print("PREFLIGHT PASSED")
    print("=" * 70)

    print(
        f"TRAIN mixtures: "
        f"{TRAIN_COUNT:,}"
    )

    print(
        f"VALIDATION mixtures: "
        f"{VAL_COUNT:,}"
    )

    print()
    print(
        "Clean speech: H3"
    )

    print(
        "Real noise: MAD + DEMAND"
    )

    print(
        "Existing RHEAR mixer: YES"
    )

    print(
        "G7 architecture changed: NO"
    )

    print(
        "RIR: NO"
    )

    print(
        "H3 speaker leakage: FIXED"
    )

    print(
        "Noise file leakage: 0"
    )

    print(
        "Noise group leakage: 0"
    )

    # =========================================================================
    # CREATE OUTPUT
    # =========================================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    train_result = (
        generate_split(
            name="train",
            count=TRAIN_COUNT,
            speech_pool=train_speech,
            noise_pool=train_noise,
            seed=SEED,
        )
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    val_result = (
        generate_split(
            name="val",
            count=VAL_COUNT,
            speech_pool=val_speech,
            noise_pool=val_noise,
            seed=SEED + 1,
        )
    )

    # =========================================================================
    # VALIDATE MANIFESTS
    # =========================================================================

    print()
    print("=" * 70)
    print("VALIDATING OUTPUT")
    print("=" * 70)

    validate_rows(
        train_result["rows"],
        TRAIN_COUNT,
        "train",
    )

    validate_rows(
        val_result["rows"],
        VAL_COUNT,
        "val",
    )

    # =========================================================================
    # VALIDATE AUDIO
    # =========================================================================

    validate_audio(
        train_result["rows"],
        "train",
    )

    validate_audio(
        val_result["rows"],
        "val",
    )

    # =========================================================================
    # FINAL GENERATED SPEAKER CHECK
    # =========================================================================

    print()
    print(
        "Checking generated speaker leakage..."
    )

    check_speech_leakage(
        train_result["rows"],
        val_result["rows"],
    )

    # =========================================================================
    # COMBINED MANIFEST
    # =========================================================================

    combined_rows = (
        train_result["rows"]
        +
        val_result["rows"]
    )

    expected_total = (
        TRAIN_COUNT
        +
        VAL_COUNT
    )

    if len(combined_rows) != expected_total:
        raise RuntimeError(
            f"Combined manifest contains "
            f"{len(combined_rows)} rows; "
            f"expected {expected_total}."
        )

    write_manifest(
        combined_rows,
        OUTPUT_ROOT
        / "manifest.jsonl",
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    write_summary(
        train_result,
        val_result,
        train_speech,
        val_speech,
        removed_speakers,
        train_noise,
        val_noise,
        test_noise,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 70)
    print("BUILD COMPLETE")
    print("=" * 70)

    print(
        f"TRAIN: "
        f"{len(train_result['rows']):,}"
    )

    print(
        f"VAL:   "
        f"{len(val_result['rows']):,}"
    )

    print(
        f"TOTAL: "
        f"{len(combined_rows):,}"
    )

    print()
    print(
        f"Output:\n"
        f"{OUTPUT_ROOT}"
    )

    print()
    print(
        "Files:"
    )

    print(
        f"  {OUTPUT_ROOT / 'train.jsonl'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'val.jsonl'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'manifest.jsonl'}"
    )

    print(
        f"  {OUTPUT_ROOT / 'dataset_summary.json'}"
    )

    print()
    print(
        "FINAL CHECKS"
    )

    print(
        "  Noise file leakage:   0"
    )

    print(
        "  Noise group leakage:  0"
    )

    print(
        "  Speaker leakage:      0"
    )

    print(
        "  Architecture changed: NO"
    )

    print()
    print(
        "READY FOR G7 TRAINING."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()