#!/usr/bin/env python3
"""INTERIM dataset for the audible A/B demo.

  clean speech : REAL RECORDINGS (LibriSpeech dev-clean, CC BY 4.0)
  noise        : SYNTHESISED by rhear.core.signals -- NOT real recordings

This exists only because the real-noise corpora (MUSAN 10.3 GB, MAD) cannot
download in time for the demo. Any metric computed on it is an INTERIM number
and is NOT a G3 result. The manifest records the provenance per sample, and the
report prints the warning, so the distinction cannot be lost downstream.
"""
import os, sys, argparse
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rhear.core import signals as sg

FS = 16_000

# Each entry produces several distinct "recordings" so the model sees variety
# within a class rather than one signal repeated.
NOISE_SPECS = {
    "engine":     [("harm", dict(f0=f, n_harm=10)) for f in (38, 45, 52, 61, 70)],
    "rotor":      [("harm", dict(f0=f, n_harm=6)) for f in (95, 110, 125, 140, 160)],
    "vehicle":    [("mix", dict(f0=f, n_harm=7, lo=120, hi=1400)) for f in (44, 55, 66, 80)],
    "wind":       [("band", dict(lo=lo, hi=hi)) for lo, hi in
                   ((140, 700), (150, 900), (180, 1200), (200, 1500))],
    "machinery":  [("band", dict(lo=lo, hi=hi)) for lo, hi in
                   ((300, 3000), (500, 4000), (250, 2200))],
    "siren":      [("siren", dict(f_lo=f, f_hi=f + 400, rate=r))
                   for f, r in ((600, 0.5), (700, 0.8), (850, 1.2))],
    "impulse":    [("impulse", dict(alpha=a, rate=r)) for a, r in
                   ((1.4, 1.2), (1.6, 0.7), (1.5, 2.0))],
}


def synth(kind, p, n, rng):
    t = np.arange(n) / FS
    if kind == "harm":
        return sg.harmonic(n, FS, rng=rng, **p)
    if kind == "band":
        return sg.broadband(n, FS, p["lo"], p["hi"], rng=rng)
    if kind == "mix":
        return (0.7 * sg.harmonic(n, FS, f0=p["f0"], n_harm=p["n_harm"], rng=rng)
                + 0.5 * sg.broadband(n, FS, p["lo"], p["hi"], rng=rng))
    if kind == "siren":
        from scipy import signal as ss
        f = p["f_lo"] + (p["f_hi"] - p["f_lo"]) * (
            0.5 + 0.5 * ss.sawtooth(2 * np.pi * p["rate"] * t, width=0.5))
        return np.sin(2 * np.pi * np.cumsum(f) / FS)
    if kind == "impulse":
        x = 0.05 * sg.broadband(n, FS, 100, 3000, rng=rng)
        for k in range(int(p["rate"] * n / FS) + 1):
            at = rng.uniform(0, n / FS - 0.05)
            x += sg.impulse_burst(n, FS, at_s=at, alpha=p["alpha"],
                                  dur_s=0.02, amp=6.0, rng=rng)
        return x
    raise ValueError(kind)


def write_noise_bank(root, seconds=12.0, rng=None):
    """One .wav per synthetic 'recording', so the split machinery can treat them
    as source recordings exactly as it treats real ones."""
    rng = rng or np.random.default_rng(11)
    n = int(seconds * FS)
    made = 0
    for cls, specs in NOISE_SPECS.items():
        d = os.path.join(root, cls)
        os.makedirs(d, exist_ok=True)
        for i, (kind, p) in enumerate(specs):
            x = synth(kind, p, n, rng)
            x = x / (np.max(np.abs(x)) + 1e-9) * 0.9
            sf.write(os.path.join(d, f"SYNTH_{cls}_{i:02d}.wav"), x.astype(np.float32), FS)
            made += 1
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300)
    a = ap.parse_args()

    rng = np.random.default_rng(11)
    nb = os.path.join(a.out, "_synth_noise")
    k = write_noise_bank(nb, rng=rng)
    print(f"  synthesised {k} noise 'recordings' across {len(NOISE_SPECS)} classes")

    # speaker-disjoint roots by symlink (LibriSpeech speaker dirs)
    spks = sorted(d for d in os.listdir(a.speech) if d.isdigit())
    rng.shuffle(spks)
    parts = {"train": spks[:26], "val": spks[26:33], "test": spks[33:]}
    roots = {}
    for split, ss in parts.items():
        d = os.path.join(a.out, "_spk_" + split)
        os.makedirs(d, exist_ok=True)
        for s in ss:
            t = os.path.join(d, s)
            if not os.path.exists(t):
                os.symlink(os.path.join(a.speech, s), t)
        roots[split] = d
        print(f"  speakers[{split}]: {len(ss)}")

    # Explicit noise index: class from the directory, provenance stated outright.
    from rhear_data import build, manifest, report, splits as S
    noise_index = []
    for cls in sorted(os.listdir(nb)):
        d = os.path.join(nb, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".wav"):
                pth = os.path.join(d, f)
                noise_index.append(dict(
                    path=pth, cls=cls, corpus="synth",
                    source=f"synth:{os.path.splitext(f)[0]}",
                    provenance="synthetically generated"))
    print(f"  noise index: {len(noise_index)} synthesised recordings, "
          f"classes {sorted({n['cls'] for n in noise_index})}")
    n, audit = build.build(
        out_dir=os.path.join(a.out, "dataset"), speech_roots=roots,
        musan_root=None, mad_root=None, rir_root=None, noise_index=noise_index,
        n_per_split={"train": a.train, "val": a.val, "test": a.test},
        dur_s=4.0, seed=11, snr_range=(-10.0, 20.0), heldout_n=0)
    print(f"\n  leakage audit: {'CLEAN' if audit['clean'] else 'LEAKAGE'}")
    hdr, rows = manifest.read_manifest(os.path.join(a.out, "dataset", "manifest.jsonl"))
    print(report.render(report.compute(hdr, rows,
                                       audio_root=os.path.join(a.out, "dataset")), hdr))
    print("\n  *** INTERIM: speech is REAL, noise is SYNTHESISED. Not a G3 result. ***")
