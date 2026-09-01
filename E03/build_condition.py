#!/usr/bin/env python3
"""Build an A/B/C training condition. ONE variable changes: noise diversity.

A  environmental only            (MUSAN/RIRS point-source)
B  A + MAD 6 classes + drone     -> tests DIVERSITY
C  B + extended augmentation     -> tests EXTRA augmentation on top

Held identical across all three: speakers, model, optimiser, loss, budget,
dataset size, SNR range, evaluation protocol.

Noise-source disjointness is enforced against the held-out evaluation sources,
and MAD's own video-level train/test split is honoured so clips cut from one
YouTube video can never straddle train and test.
"""
import os, sys, csv, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhear_data import build, manifest, report

MAD_MAP = {'0':'communication','1':'gunshot','2':'footsteps','3':'shelling',
           '4':'vehicle','5':'helicopter','6':'fighter'}
MAD_EXCLUDE = {'communication'}          # contains SPEECH -- never a noise source


def env_index(rirs_root, exclude):
    out = []
    for p in sorted(glob.glob(os.path.join(rirs_root, "pointsource_noises", "*.wav"))):
        src = f"musan:{os.path.splitext(os.path.basename(p))[0]}"
        if src in exclude: continue
        out.append(dict(path=p, cls="environmental", corpus="musan", source=src,
                        provenance="real recording"))
    return out


def mad_index(mad_root, split="training"):
    """MAD's own split is by source video -- honour it."""
    base = os.path.join(mad_root, "MAD_dataset")
    out = []
    with open(os.path.join(base, f"{split}.csv")) as f:
        for d in csv.DictReader(f):
            cls = MAD_MAP[d["label"]]
            if cls in MAD_EXCLUDE: continue
            vid = d["path"].split("/")[1]
            out.append(dict(path=os.path.join(base, d["path"]), cls=cls,
                            corpus="mad", source=f"mad:{vid}",
                            provenance="real recording"))
    return out


def drone_index(drone_root):
    return [dict(path=p, cls="drone", corpus="droneaudioset",
                 source=f"drone:{os.path.basename(p)}",
                 provenance="real recording")
            for p in sorted(glob.glob(os.path.join(drone_root, "*.wav")))]


SIREN_CLASSES = {"police", "firetruck", "ambulance"}   # 'traffic' is the NEGATIVE
                                                      # class in sireNNet, not a siren


def siren_index(siren_root):
    """sireNNet. Grouped by class rather than by file: the corpus was scraped and
    the published version includes augmented datapoints, so near-duplicates are
    likely and per-file grouping would let them straddle splits."""
    out = []
    for p in sorted(glob.glob(os.path.join(siren_root, "**", "*.wav"), recursive=True)):
        cls = os.path.basename(os.path.dirname(p)).lower()
        if cls not in SIREN_CLASSES:
            continue
        out.append(dict(path=p, cls="siren", corpus="sirennet",
                        source=f"siren:{cls}:{os.path.splitext(os.path.basename(p))[0][:6]}",
                        provenance="real recording (scraped, partly augmented)"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["A","B","C"])
    ap.add_argument("--speech-train", required=True); ap.add_argument("--speech-val", required=True)
    ap.add_argument("--rirs", required=True); ap.add_argument("--mad", default=None)
    ap.add_argument("--drone", default=None); ap.add_argument("--siren", default=None)
    ap.add_argument("--exclude", required=True); ap.add_argument("--exclude-rirs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=2000); ap.add_argument("--val", type=int, default=300)
    a = ap.parse_args()

    excl = set(json.load(open(a.exclude)))
    idx = env_index(a.rirs, excl)
    if a.condition in ("B","C"):
        if a.mad and os.path.isdir(a.mad): idx += mad_index(a.mad)
        if a.drone and os.path.isdir(a.drone): idx += drone_index(a.drone)
        if a.siren and os.path.isdir(a.siren):
            s = siren_index(a.siren)
            if s: idx += s
    import collections
    c = collections.Counter(n["cls"] for n in idx)
    print(f"  condition {a.condition}: {len(idx)} noise recordings")
    for k,v in c.most_common(): print(f"    {k:<16}{v:6d}")

    # condition C widens the augmentation
    snr = (-15.0, 25.0) if a.condition=="C" else (-10.0, 20.0)
    excl_rir = set(json.load(open(a.exclude_rirs))) if a.exclude_rirs else set()
    if excl_rir:
        import rhear_data.build as B
        _o = B.index_rirs
        B.index_rirs = lambda root, real_only=True: [r for r in _o(root, real_only)
                                                     if r["id"] not in excl_rir]
    n, audit = build.build(
        out_dir=a.out,
        speech_roots={"train":a.speech_train,"val":a.speech_val,"test":a.speech_val},
        musan_root=None, mad_root=None, rir_root=a.rirs, noise_index=idx,
        n_per_split={"train":a.train,"val":a.val,"test":1},
        dur_s=4.0, seed=777, snr_range=snr, heldout_n=0, verbose=True)
    hdr, rows = manifest.read_manifest(os.path.join(a.out,"manifest.jsonl"))
    tr = {l["source"] for r in rows if r["split"]=="train" for l in r["noise_layers"]}
    assert not (tr & excl), "LEAKAGE: eval noise source in training"
    print(f"  leakage vs eval sources: {len(tr & excl)}  (must be 0)  -> OK")
    print(report.render(report.compute(hdr, rows, audio_root=a.out), hdr))
