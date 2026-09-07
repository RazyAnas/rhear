#!/usr/bin/env python3
"""Build the G11 dataset: same clip count as h3_20k, 9.7x the speakers.

h3_20k drew 20,000 training clips from 26 SPEAKERS (LibriSpeech dev-clean +
test-clean were used as the train pool). Every architecture lever tested since
-- capacity (G9), resolution (G8), objective (G10) -- moved the TRAIN-set fit by
<= 0.02 PESQ, while the train-val gap stayed at ~0.30. Speaker diversity is the
untested variable.

This build changes ONE thing: the train speaker pool becomes train-clean-100
(251 speakers), with dev-clean as val and test-clean as test, which are
speaker-disjoint by LibriSpeech's own construction. Clip count is held at 20,000
so the comparison against h3_20k isolates speaker diversity.

MAD on the Kaggle mirror needs special handling: its layout is
MAD_dataset/{training,test}/<video_id>/<n>.wav with classes only in a CSV, so
the directory-based index_mad() cannot see them and excluded 0 'communication'
clips. Label 0 is communication (speech) -- confirmed both by the documented
class order and by measured 2-8 Hz syllabic modulation (0.413 vs 0.19-0.33 for
every other label). Those 981 clips are excluded here explicitly; letting speech
into the noise pool would corrupt the enhancement target.
"""
import csv, os, sys
sys.path.insert(0, os.path.expanduser("~/PS#2/handoff/code"))

from rhear_data.build import build, index_musan_noise, index_rirs, _is_junk
from rhear_data import splits as S

C = "/Volumes/RHEARDATA/corpora"
OUT = os.path.expanduser("~/PS#2/handoff/data/g11_20k")
MAD_COMMUNICATION = "0"          # verified: speech

def mad_noise():
    """MAD from the CSVs, excluding the speech class, grouped by source video."""
    out, skipped = [], 0
    root = f"{C}/mad/MAD_dataset"
    for csvf in ("training.csv", "test.csv"):
        for r in list(csv.reader(open(os.path.join(root, csvf))))[1:]:
            if len(r) < 3: continue
            rel, lab = r[1], r[2]
            if lab == MAD_COMMUNICATION:
                skipped += 1; continue
            p = os.path.join(root, rel)
            if not os.path.exists(p): continue
            vid = os.path.basename(os.path.dirname(rel))     # the source video
            out.append(dict(path=p, cls=f"mad_{lab}", source=f"mad:{vid}",
                            corpus="mad", sub=f"label{lab}",
                            provenance="real recording"))
    return out, skipped

def noisex():
    """NOISEX-92 + Noise15 + Nonspeech: ~130 extra noise TYPES."""
    out = []
    for dp, _, fns in os.walk(f"{C}/Noises"):
        for fn in fns:
            if not fn.lower().endswith(".wav") or _is_junk(fn): continue
            p = os.path.join(dp, fn)
            out.append(dict(path=p, cls="military_bench", corpus="noisex",
                            source=f"noisex:{os.path.splitext(fn)[0]}",
                            sub=os.path.basename(dp), provenance="real recording"))
    return out

def main():
    mus = index_musan_noise(f"{C}/musan")
    mad, skipped = mad_noise()
    nx  = noisex()
    noise = mus + mad + nx
    print(f"noise pool: {len(mus)} MUSAN + {len(mad)} MAD + {len(nx)} NOISEX/etc "
          f"= {len(noise)}   ({skipped} MAD 'communication' speech clips EXCLUDED)")
    print(f"noise source groups: {len({n['source'] for n in noise})}")

    speech_roots = {"train": f"{C}/LibriSpeech/train-clean-100",
                    "val":   f"{C}/LibriSpeech/dev-clean",
                    "test":  f"{C}/LibriSpeech/test-clean"}
    n, rep = build(OUT, speech_roots, None, None, f"{C}/RIRS_NOISES",
                   n_per_split={"train": 20000, "val": 500, "test": 500},
                   dur_s=4.0, seed=777, snr_range=(-10.0, 20.0),
                   heldout_n=0, noise_index=noise)
    print(f"\nwrote {n} samples -> {OUT}")
    print(f"leakage: {'CLEAN' if rep['clean'] else 'LEAKAGE DETECTED'}")

if __name__ == "__main__":
    main()
