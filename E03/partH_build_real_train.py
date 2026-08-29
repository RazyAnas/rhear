#!/usr/bin/env python3
"""Build the REAL-noise training set, excluding the real-noise TEST sources.

The partG test noise came from RIRS_NOISES/pointsource_noises, which are
themselves extracted from MUSAN. Training on full MUSAN without excluding them
would put the same recordings in train and test -- invisible leakage that would
make the whole comparison worthless.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhear_data import build, manifest, report, splits as S

ap = argparse.ArgumentParser()
ap.add_argument("--speech-train", required=True)
ap.add_argument("--speech-val", required=True)
ap.add_argument("--musan-noise", required=True)
ap.add_argument("--rirs", required=True)
ap.add_argument("--exclude", required=True)
ap.add_argument("--exclude-rirs", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--train", type=int, default=2000)
ap.add_argument("--val", type=int, default=300)
a = ap.parse_args()

excl = set(json.load(open(a.exclude)))
noise_index, skipped = [], 0
for d, _, fs in os.walk(a.musan_noise):
    for f in sorted(fs):
        if not f.endswith(".wav"):
            continue
        src = f"musan:{os.path.splitext(f)[0]}"
        if src in excl:
            skipped += 1
            continue
        noise_index.append(dict(path=os.path.join(d, f), cls="environmental",
                                corpus="musan", source=src,
                                provenance="real recording"))
print(f"  real noise: {len(noise_index)} files usable, {skipped} excluded "
      f"(they are in the real-noise TEST set)")
assert skipped > 0, "expected to exclude the partG test sources - check the naming"

# RIR rooms used by the real-noise TEST set must also be held out, or the model
# sees the same measured room in training and evaluation.
excl_rir = set(json.load(open(a.exclude_rirs))) if a.exclude_rirs else set()
if excl_rir:
    import rhear_data.build as B
    _orig = B.index_rirs
    def _filtered(root, real_only=True):
        out = [r for r in _orig(root, real_only) if r["id"] not in excl_rir]
        print(f"  RIRs: {len(out)} usable, {len(_orig(root, real_only))-len(out)} excluded "
              f"(used by the real-noise TEST set)")
        return out
    B.index_rirs = _filtered

roots = {"train": a.speech_train, "val": a.speech_val, "test": a.speech_val}
n, audit = build.build(
    out_dir=a.out, speech_roots=roots, musan_root=None, mad_root=None,
    rir_root=a.rirs, noise_index=noise_index,
    n_per_split={"train": a.train, "val": a.val, "test": 1},
    dur_s=4.0, seed=777, snr_range=(-10.0, 20.0), heldout_n=0, verbose=True)

hdr, rows = manifest.read_manifest(os.path.join(a.out, "manifest.jsonl"))
tr = {l["source"] for r in rows if r["split"] == "train" for l in r["noise_layers"]}
print(f"\n  train noise sources: {len(tr)}")
print(f"  overlap with real TEST sources: {len(tr & excl)}  (MUST be 0)")
assert not (tr & excl), "LEAKAGE: a test noise source appears in training"
trr = {r["rir_id"] for r in rows if r["split"] == "train" and r.get("rir_id")}
print(f"  train RIRs {len(trr)}, overlap with test RIRs: {len(trr & excl_rir)}  (MUST be 0)")
assert not (trr & excl_rir), "LEAKAGE: a test RIR appears in training"
print(report.render(report.compute(hdr, rows, audio_root=a.out), hdr))
