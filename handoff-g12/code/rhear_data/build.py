"""Build the RHEAR E03 dataset from real corpora.

Runs identically on a laptop and in Colab. The only difference is where the
corpora live, which is a path argument.
"""
import json
import os
import numpy as np
import soundfile as sf
from . import mixing, splits as S
from .manifest import ManifestWriter, build_header
from .registry import DATASETS, derived_licence

FS = 16_000


# ------------------------------------------------------------------ indexing
def index_librispeech(root):
    """-> {speaker: [(utt_id, path), ...]}  (REAL recordings)"""
    out = {}
    for d, _, fs in os.walk(root, followlinks=True):   # split roots may be symlinks
        for f in fs:
            if f.endswith((".flac", ".wav")):
                p = os.path.join(d, f)
                spk, utt = S.librispeech_speaker(p)
                if spk:
                    out.setdefault(spk, []).append((utt, p))
    return out


def index_musan_noise(root):
    """MUSAN noise subtree only. -> [dict(path, cls, source)]

    If the tree contains a directory literally named `noise` (the MUSAN layout)
    only that subtree is indexed, so music/ and speech/ can never leak in as
    "noise". If it does not, `root` is assumed to already point at noise.
    """
    walk = list(os.walk(root, followlinks=True))
    has_noise_dir = any("noise" in d.replace(os.sep, "/").split("/") for d, _, _ in walk)
    out = []
    for d, _, fs in walk:
        if has_noise_dir and "noise" not in d.replace(os.sep, "/").split("/"):
            continue
        for f in fs:
            if not f.endswith(".wav"):
                continue
            p = os.path.join(d, f)
            sub = "free-sound" if "free-sound" in d else (
                "sound-bible" if "sound-bible" in d else "other")
            out.append(dict(path=p, cls="ambient" if sub == "free-sound" else "machinery",
                            source=S.noise_group(p, "musan"), corpus="musan", sub=sub,
                            provenance=("synthetically generated"
                                        if "STANDIN" in p else "real recording")))
    return out


MAD_EXCLUDE = {"communication"}          # contains speech -- never a noise source


def index_mad(root):
    """MAD as a MILITARY-NOISE SOURCE ONLY. The 'communication' class contains
    speech and is excluded so it cannot contaminate the enhancement target."""
    out, skipped = [], 0
    for d, _, fs in os.walk(root, followlinks=True):
        cls = os.path.basename(d).lower()
        for f in fs:
            if not f.endswith((".wav", ".flac")):
                continue
            if cls in MAD_EXCLUDE:
                skipped += 1
                continue
            p = os.path.join(d, f)
            out.append(dict(path=p, cls=cls, source=S.noise_group(p, "mad"),
                            corpus="mad", sub=cls,
                            provenance=("synthetically generated"
                                        if "STANDIN" in p else "real recording")))
    return out, skipped


def index_rirs(root, real_only=True):
    out = []
    for d, _, fs in os.walk(root, followlinks=True):
        dl = d.replace(os.sep, "/").lower()
        is_real = "real_rirs" in dl
        if real_only and not is_real:
            continue
        for f in fs:
            if f.endswith(".wav"):
                p = os.path.join(d, f)
                out.append(dict(path=p, id=os.path.splitext(f)[0],
                                kind="real measured" if is_real else "simulated",
                                room=S.rir_group(p)))
    return out


# ------------------------------------------------------------------ io
class UnreadableAudio(Exception):
    pass


def load_audio(path, fs=FS):
    """Public corpora contain occasional corrupt files (LibriSpeech dev-clean has
    exactly one that fails to decode: 3853-163249-0052.flac). A build must not
    die on one bad file, but it must not silently swallow it either -- the count
    goes into the manifest header."""
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as ex:
        raise UnreadableAudio(f"{path}: {ex}") from ex
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != fs:
        from scipy import signal as sg
        g = np.gcd(int(sr), int(fs))
        x = sg.resample_poly(x, fs // g, sr // g).astype(np.float32)
    return x


# ------------------------------------------------------------------ build
def build(out_dir, speech_roots, musan_root, mad_root, rir_root,
          n_per_split=None, dur_s=4.0, seed=1234, snr_range=(-10.0, 20.0),
          heldout_n=200, verbose=True, noise_index=None):
    """speech_roots: {"train": path, "val": path, "test": path}

    noise_index: optional explicit list of noise entries, each a dict with
    path/cls/source/corpus/provenance. Provided rather than inferred so a caller
    cannot accidentally mislabel provenance -- inferring it from filenames
    silently marked synthesised noise as "real recording" once, which is the one
    mistake this field exists to prevent.
    """
    n_per_split = n_per_split or {"train": 2000, "val": 250, "test": 250}
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    unreadable = set()

    speech = {k: index_librispeech(v) for k, v in speech_roots.items()}
    if noise_index is not None:
        noise, mad_skipped = list(noise_index), 0
        missing = [n for n in noise if "provenance" not in n]
        if missing:
            raise ValueError("every noise entry must state its provenance "
                             f"explicitly; {len(missing)} do not")
    else:
        noise = index_musan_noise(musan_root) if musan_root else []
        mad, mad_skipped = index_mad(mad_root) if mad_root else ([], 0)
        noise = noise + mad
    rirs = index_rirs(rir_root) if rir_root else []
    if verbose:
        for k, v in speech.items():
            print(f"  speech[{k}]: {len(v)} speakers, "
                  f"{sum(len(x) for x in v.values())} utterances")
        print(f"  noise: {len(noise)} clips "
              f"({sum(1 for n in noise if n['corpus']=='musan')} MUSAN, "
              f"{sum(1 for n in noise if n['corpus']=='mad')} MAD; "
              f"{mad_skipped} MAD 'communication' clips excluded)")
        print(f"  rirs: {len(rirs)} real measured")

    # noise and rooms split by SOURCE GROUP, never by clip
    ngroups = S.split_groups([n["source"] for n in noise],
                             np.random.default_rng(seed + 1))
    rgroups = S.split_groups([r["room"] for r in rirs],
                             np.random.default_rng(seed + 2)) if rirs else \
        {k: set() for k in S.SPLITS}
    noise_by = {s: [n for n in noise if n["source"] in ngroups[s]] for s in S.SPLITS}
    rir_by = {s: [r for r in rirs if r["room"] in rgroups[s]] for s in S.SPLITS}

    header = build_header(
        dict(seed=seed, fs=FS, dur_s=dur_s, snr_range=list(snr_range),
             n_per_split=n_per_split, heldout_n=heldout_n),
        {k: dict(name=DATASETS[k].name, licence=DATASETS[k].licence,
                 content_kind=DATASETS[k].content_kind)
         for k in ("librispeech_dev_clean", "librispeech_test_clean",
                   "musan_noise", "mad", "rirs_noises") if k in DATASETS})
    header["derived_licence"] = derived_licence(
        ["librispeech_dev_clean", "musan_noise", "mad", "rirs_noises"])
    mw = ManifestWriter(os.path.join(out_dir, "manifest.jsonl"), header)
    audit = S.LeakageAudit()

    plan = [(s, n_per_split[s]) for s in S.SPLITS] + [("heldout", heldout_n)]
    for split, count in plan:
        src = "test" if split == "heldout" else split
        spk_pool = speech.get(src, {})
        nz_pool = noise_by[src]
        rir_pool = rir_by[src]
        if not spk_pool or not nz_pool:
            raise RuntimeError(
                f"split '{split}' has no data (speakers={len(spk_pool)}, "
                f"noise groups={len(nz_pool)}). A silently skipped split is worse "
                f"than a failed build -- it produces a dataset that looks complete.")
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        spks = sorted(spk_pool)
        for i in range(count):
            for attempt in range(8):
                spk = spks[int(rng.integers(len(spks)))]
                utt, ppath = spk_pool[spk][int(rng.integers(len(spk_pool[spk])))]
                try:
                    sp = dict(audio=load_audio(ppath), speaker=spk, utt=utt)
                    break
                except UnreadableAudio as ex:
                    unreadable.add(str(ex).split(":")[0])
            else:
                raise RuntimeError("8 consecutive unreadable speech files -- "
                                   "the corpus is probably damaged, not just "
                                   "missing one file")
            k = int(rng.integers(1, 4))
            picks = [nz_pool[int(rng.integers(len(nz_pool)))] for _ in range(k)]
            nz = [dict(audio=load_audio(p["path"]), cls=p["cls"],
                       source=p["source"],
                       provenance=p.get("provenance", "real recording"))
                  for p in picks]
            rir = None
            if rir_pool and rng.random() < 0.6:
                rp = rir_pool[int(rng.integers(len(rir_pool)))]
                rir = dict(audio=load_audio(rp["path"]), id=rp["id"], kind=rp["kind"])
                room = rp["room"]
            else:
                room = None
            noisy, clean, meta = mixing.make_mixture(
                sp, nz, FS, rng, rir=rir, dur_s=dur_s, snr_range=snr_range)
            base = f"{split}_{i:06d}"
            sf.write(os.path.join(d, base + "_noisy.wav"), noisy, FS)
            sf.write(os.path.join(d, base + "_clean.wav"), clean, FS)
            meta.update(id=base, split=split, heldout=(split == "heldout"),
                        noisy=f"{split}/{base}_noisy.wav",
                        clean=f"{split}/{base}_clean.wav",
                        speech_source_path=os.path.relpath(ppath, speech_roots[src]))
            mw.add(meta)
            audit.add(split if split != "heldout" else "test", speaker=spk,
                      noise_groups=[p["source"] for p in picks], room=room)
            if verbose and (i + 1) % 250 == 0:
                print(f"    {split}: {i+1}/{count}")
    n = mw.close()
    if unreadable:
        print(f"  NOTE: skipped {len(unreadable)} unreadable source file(s): "
              f"{sorted(unreadable)[:3]}")
    rep = audit.report()
    rep["unreadable_sources"] = sorted(unreadable)
    with open(os.path.join(out_dir, "leakage_audit.json"), "w") as f:
        json.dump(rep, f, indent=2)
    if verbose:
        print(f"  wrote {n} samples -> {out_dir}")
        print(f"  leakage audit: {'CLEAN' if rep['clean'] else 'LEAKAGE DETECTED'}")
    return n, rep
