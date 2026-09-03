from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "RAW"
SPLITS = ROOT / "SPLITS"

SEED = 42
random.seed(SEED)

rows = []

# --------------------------------------------------
# DEMAND
# Split by environment, NOT individual WAV files
# --------------------------------------------------

demand = RAW / "DEMAND"

if demand.exists():
    for group in sorted(demand.iterdir()):
        if not group.is_dir():
            continue

        wavs = list(group.rglob("*.wav"))

        if not wavs:
            continue

        for wav in wavs:
            rows.append({
                "path": wav.relative_to(RAW).as_posix(),
                "dataset": "DEMAND",
                "source_group": group.name,
                "split": None,
            })

# --------------------------------------------------
# Show groups
# --------------------------------------------------

groups = {}

for row in rows:
    groups.setdefault(row["source_group"], []).append(row)

print("\nSOURCE GROUPS:")
for group, items in groups.items():
    print(f"{group}: {len(items)} files")

# --------------------------------------------------
# IMPORTANT:
# Current DEMAND has only a small number of groups.
#
# For now use deterministic group assignment.
# We will redo this once MAD/FSD50K/ASR are present.
# --------------------------------------------------

group_names = sorted(groups.keys())

random.shuffle(group_names)

n = len(group_names)

if n >= 5:
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)

    # Make sure test gets at least one group
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    train_groups = set(group_names[:n_train])
    val_groups = set(group_names[n_train:n_train+n_val])
    test_groups = set(group_names[n_train+n_val:])
else:
    print("\nWARNING:")
    print("Not enough independent DEMAND groups for a meaningful")
    print("70/15/15 split.")
    print("Creating a temporary deterministic split for testing.")

    train_groups = set(group_names[:-2])
    val_groups = set(group_names[-2:-1])
    test_groups = set(group_names[-1:])

# --------------------------------------------------
# Assign split
# --------------------------------------------------

for row in rows:
    group = row["source_group"]

    if group in train_groups:
        row["split"] = "train"
    elif group in val_groups:
        row["split"] = "validation"
    elif group in test_groups:
        row["split"] = "test"

# --------------------------------------------------
# Write CSVs
# --------------------------------------------------

SPLITS.mkdir(exist_ok=True)

for split in ["train", "validation", "test"]:

    output = SPLITS / f"{split}.csv"

    split_rows = [
        row for row in rows
        if row["split"] == split
    ]

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "dataset",
                "source_group",
                "split",
            ],
        )

        writer.writeheader()
        writer.writerows(split_rows)

    print(
        f"{split:12s}: "
        f"{len(split_rows):4d} files -> {output}"
    )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nGROUP ASSIGNMENT:")

print("TRAIN:")
for g in sorted(train_groups):
    print("  ", g)

print("VALIDATION:")
for g in sorted(val_groups):
    print("  ", g)

print("TEST:")
for g in sorted(test_groups):
    print("  ", g)

print("\nDone.")