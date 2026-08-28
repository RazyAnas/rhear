#!/usr/bin/env python3
"""PART G - build a REAL-NOISE evaluation set and measure synthetic -> real drop.

  speech : REAL LibriSpeech, the SAME 7 held-out test speakers as the frozen
           evaluation set, so the speech side is unchanged.
  noise  : REAL recordings -- MUSAN point-source noises shipped inside
           RIRS_NOISES (843 files, 5.91 h, 16 kHz).
  rooms  : REAL measured RIRs (RWCP / REVERB-2014 / AIR), 417 responses.

The frozen model trained on 100% synthetic noise, so every noise source here is
unseen by construction -- there is no leakage to audit on the noise side.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhear_data import build, manifest, report, splits as S

ap = argparse.ArgumentParser()
ap.add_argument("--speech", required=True)
ap.add_argument("--rirs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

noise_dir = os.path.join(a.rirs, "pointsource_noises")
noise_index = []
for f in sorted(os.listdir(noise_dir)):
    if f.endswith(".wav"):
        p = os.path.join(noise_dir, f)
        noise_index.append(dict(
            path=p, cls="environmental", corpus="musan_freesound",
            source=f"musan:{os.path.splitext(f)[0]}",
            provenance="real recording"))
print(f"  real noise recordings: {len(noise_index)}")

os.makedirs(a.out, exist_ok=True)
roots = {k: a.speech for k in ("train", "val", "test")}
n, audit = build.build(
    out_dir=a.out, speech_roots=roots,
    musan_root=None, mad_root=None,
    rir_root=a.rirs, noise_index=noise_index,
    n_per_split={"train": 1, "val": 1, "test": a.n},
    dur_s=4.0, seed=4242, snr_range=(-10.0, 20.0), heldout_n=0, verbose=True)

hdr, rows = manifest.read_manifest(os.path.join(a.out, "manifest.jsonl"))
test = [r for r in rows if r["split"] == "test"]
prov = {}
for r in test:
    for l in r["noise_layers"]:
        prov[l["provenance"]] = prov.get(l["provenance"], 0) + 1
rir = {}
for r in test:
    rir[r["rir_kind"]] = rir.get(r["rir_kind"], 0) + 1
print(f"\n  REAL-NOISE TEST SET: {len(test)} clips")
print(f"    speakers        {len({r['speaker'] for r in test})}")
print(f"    noise sources   {len({l['source'] for r in test for l in r['noise_layers']})}")
print(f"    provenance      {prov}")
print(f"    reverberation   {rir}")
