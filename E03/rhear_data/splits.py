"""Split construction and leakage auditing.

Three independent disjointness requirements, all enforced at the SOURCE
RECORDING level rather than the clip level -- otherwise two clips cut from one
original recording can land in different splits and leak in disguised form.

  speakers : LibriSpeech's own splits are already speaker-disjoint, so
             train-clean-100 / dev-clean / test-clean map straight onto
             train / val / test. The audit verifies it rather than assuming it.
  noise    : noise files are grouped by SOURCE recording and the groups are
             split. For MAD, clips are grouped by their source video where the
             filename exposes it; where it does not, that is reported as a
             residual risk instead of being hidden.
  rooms    : RIRs are grouped by room so the same room never appears in two
             splits.

The RHEAR REAL-WORLD TEST SET is built from the test split and is never used for
training or tuning. It is written to a separate directory and its manifest is
marked `heldout=True`.
"""
import hashlib
import os
import re
from collections import defaultdict

SPLITS = ("train", "val", "test")


def librispeech_speaker(path):
    """LibriSpeech layout: <split>/<speaker>/<chapter>/<spk>-<chap>-<utt>.flac"""
    m = re.search(r"/(\d+)/(\d+)/\1-\2-(\d+)\.", path.replace(os.sep, "/"))
    return (m.group(1), f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else (None, None)


def noise_group(path, corpus):
    """Group key identifying the SOURCE recording a clip came from."""
    b = os.path.basename(path)
    if corpus == "musan":
        # MUSAN noise files are individual recordings: the file IS the group.
        return f"musan:{os.path.splitext(b)[0]}"
    if corpus == "mad":
        # MAD clips are segmented from source videos. Where the filename carries
        # a video/source id, group on it; otherwise fall back to the clip and
        # record the residual risk.
        m = re.match(r"([A-Za-z0-9_\-]+?)[_\-]\d+\.(wav|flac)$", b)
        return f"mad:{m.group(1)}" if m else f"mad_clip:{os.path.splitext(b)[0]}"
    return f"{corpus}:{os.path.splitext(b)[0]}"


def describe_grouping(paths, corpus):
    """Report how clips collapse into source groups.

    MUST be inspected against the real corpus before trusting the split. If MAD
    filenames do not expose the source video, every clip of a class can collapse
    into ONE group (seen with stand-in files: all gunshots -> "mad:gunshot"),
    which would push a whole class into a single split. The opposite failure --
    one group per clip -- means clips from the same source video can land in
    different splits and leak.
    """
    from collections import Counter
    g = [noise_group(p, corpus) for p in paths]
    c = Counter(g)
    return {
        "clips": len(paths),
        "groups": len(c),
        "clips_per_group_mean": round(len(paths) / max(len(c), 1), 2),
        "largest_group": c.most_common(1)[0] if c else None,
        "singleton_groups": sum(1 for v in c.values() if v == 1),
        "risk": ("groups collapse a whole class -- check the filename pattern"
                 if len(c) < 5 else
                 "one group per clip -- clips from one source video may leak"
                 if all(v == 1 for v in c.values()) and len(c) > 50 else "looks sane"),
    }


def rir_group(path):
    """Group RIRs by room directory, not by individual measurement."""
    p = path.replace(os.sep, "/")
    m = re.search(r"(Room\d+|room\d+|air_[a-z_]+|RVB\d+|[A-Za-z]+_room)", p)
    return m.group(1) if m else os.path.dirname(p)


def split_groups(groups, rng, frac=(0.80, 0.10, 0.10)):
    """Deterministically assign whole groups to splits.

    Guarantees at least one group per split whenever there are >= 3 groups.
    Without that, a small noise pool silently produced an EMPTY validation split
    (4 groups x 10% rounds to 0), and the build skipped `val` entirely.
    """
    g = sorted(set(groups))
    rng.shuffle(g)
    n = len(g)
    if n < 3:
        return {"train": set(g), "val": set(), "test": set()}
    a = max(1, int(frac[0] * n))
    b = max(a + 1, int((frac[0] + frac[1]) * n))
    b = min(b, n - 1)
    a = min(a, b - 1) if b - 1 >= 1 else a
    return {"train": set(g[:a]), "val": set(g[a:b]), "test": set(g[b:])}


def file_hash(path, nbytes=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


class LeakageAudit:
    """Checks that nothing appears in more than one split, by identity and by
    content hash."""

    def __init__(self):
        self.speakers = defaultdict(set)
        self.noise = defaultdict(set)
        self.rooms = defaultdict(set)
        self.hashes = defaultdict(set)

    def add(self, split, speaker=None, noise_groups=(), room=None, hashes=()):
        if speaker:
            self.speakers[split].add(speaker)
        for g in noise_groups:
            self.noise[split].add(g)
        if room:
            self.rooms[split].add(room)
        for h in hashes:
            self.hashes[split].add(h)

    def _pairs(self, d):
        out = {}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                ov = d.get(a, set()) & d.get(b, set())
                if ov:
                    out[f"{a}|{b}"] = sorted(ov)[:10]
        return out

    def report(self):
        r = {
            "speaker_overlap": self._pairs(self.speakers),
            "noise_group_overlap": self._pairs(self.noise),
            "room_overlap": self._pairs(self.rooms),
            "content_hash_overlap": self._pairs(self.hashes),
            "counts": {s: dict(speakers=len(self.speakers.get(s, ())),
                               noise_groups=len(self.noise.get(s, ())),
                               rooms=len(self.rooms.get(s, ())))
                       for s in SPLITS},
        }
        r["clean"] = not any(r[k] for k in
                             ("speaker_overlap", "noise_group_overlap",
                              "room_overlap", "content_hash_overlap"))
        return r
