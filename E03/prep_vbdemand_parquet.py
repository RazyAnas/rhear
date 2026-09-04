#!/usr/bin/env python3
"""Unpack the HuggingFace VoiceBank+DEMAND-16k parquet shards to WAV pairs.

The Edinburgh datashare host stopped serving mid-download (it went from working
to returning zero bytes), so the corpus comes from
`JacobLinCool/VoiceBank-DEMAND-16k` instead. That copy is already at 16 kHz,
which removes the 48 -> 16 kHz resampling step entirely.

Output matches what prep_vbdemand.py produced, so finetune_vbdemand.py and
eval_vbdemand.py run against it unchanged.

TWO SPEAKERS ARE HELD OUT FOR VALIDATION. The 824-utterance test set is not
touched here — checkpoint selection happens on held-out TRAIN speakers and the
test set is scored exactly once, at the end. Selecting on the test set is how a
benchmark number stops meaning anything.

    /opt/anaconda3/bin/python E03/prep_vbdemand_parquet.py
"""
import os, io, sys, json, glob, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq

FS = 16000
VAL_SPEAKERS = {"p286", "p287"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/tmp/vbhf")
ap.add_argument("--out", default="/tmp/vbhf/prep16k")
a = ap.parse_args()

shards = sorted(glob.glob(os.path.join(a.src, "train-*.parquet")))
if not shards:
    sys.exit(f"no train parquet shards under {a.src}")
print(f"\nshards {len(shards)}")

os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
os.makedirs(os.path.join(a.out, "val"), exist_ok=True)
rows = {"train": [], "val": []}
n = 0
bad = 0

for sh in shards:
    t = pq.read_table(sh)
    ids = t["id"].to_pylist()
    clean = t["clean"].to_pylist()
    noisy = t["noisy"].to_pylist()
    for i, uid in enumerate(ids):
        spk = uid.split("_")[0]
        split = "val" if spk in VAL_SPEAKERS else "train"
        cp = os.path.join(a.out, split, f"{uid}_clean.wav")
        np_ = os.path.join(a.out, split, f"{uid}_noisy.wav")
        if not (os.path.exists(cp) and os.path.exists(np_)):
            try:
                c, sr1 = sf.read(io.BytesIO(clean[i]["bytes"]), dtype="float32")
                z, sr2 = sf.read(io.BytesIO(noisy[i]["bytes"]), dtype="float32")
            except Exception:
                bad += 1
                continue
            if sr1 != FS or sr2 != FS:
                bad += 1
                continue
            if c.ndim > 1:
                c = c.mean(1)
            if z.ndim > 1:
                z = z.mean(1)
            L = min(len(c), len(z))
            sf.write(cp, c[:L], FS)
            sf.write(np_, z[:L], FS)
        rows[split].append({"id": uid, "speaker": spk,
                            "clean": os.path.relpath(cp, a.out),
                            "noisy": os.path.relpath(np_, a.out)})
        n += 1
        if n % 1000 == 0:
            print(f"\r  {n} pairs", end="", flush=True)
    del t, clean, noisy
print()

with open(os.path.join(a.out, "manifest.json"), "w") as f:
    json.dump({"fs": FS, "val_speakers": sorted(VAL_SPEAKERS),
               "n_train": len(rows["train"]), "n_val": len(rows["val"]),
               "source": "JacobLinCool/VoiceBank-DEMAND-16k",
               "rows": rows}, f)

spk = sorted({r["speaker"] for r in rows["train"]})
print(f"\ntrain  {len(rows['train'])} pairs, {len(spk)} speakers")
print(f"val    {len(rows['val'])} pairs, {sorted(VAL_SPEAKERS)}")
if bad:
    print(f"skipped {bad} unreadable rows")
print(f"\nwritten {a.out}/manifest.json")
