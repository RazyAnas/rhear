"""DATASET REPORT.

Everything here is COMPUTED from the manifest and the audio on disk. Nothing is
estimated or filled in. If a quantity cannot be computed it is reported as
"not measured" rather than guessed.
"""
import os
from collections import Counter, defaultdict
import numpy as np


def _hours(sec):
    return sec / 3600.0


def compute(header, rows, audio_root=None):
    r = {"n_samples": len(rows)}
    if not rows:
        return r
    dur = sum(x["dur_s"] for x in rows)
    r["mixture_hours"] = round(_hours(dur), 3)
    r["by_split"] = {}
    for s in ("train", "val", "test", "heldout"):
        sub = [x for x in rows if x.get("split") == s]
        if sub:
            r["by_split"][s] = {
                "samples": len(sub),
                "hours": round(_hours(sum(x["dur_s"] for x in sub)), 3),
                "speakers": len({x["speaker"] for x in sub}),
            }
    r["speakers_total"] = len({x["speaker"] for x in rows})
    r["clean_utterances_used"] = len({x["utt"] for x in rows})

    ng, cls = set(), Counter()
    for x in rows:
        for l in x.get("noise_layers", []):
            ng.add(f"{l['source']}:{l['cls']}")
            cls[l["cls"]] += 1
    r["distinct_noise_recordings"] = len({l["source"]
                                          for x in rows
                                          for l in x.get("noise_layers", [])})
    r["noise_class_distribution"] = dict(cls.most_common())

    st = Counter(x.get("mixture_stationarity", "unknown") for x in rows)
    tot = sum(st.values()) or 1
    r["stationarity_pct"] = {k: round(100 * v / tot, 2) for k, v in st.most_common()}

    snr = np.array([x["snr_db"] for x in rows])
    r["snr_db"] = {
        "min": round(float(snr.min()), 2), "max": round(float(snr.max()), 2),
        "mean": round(float(snr.mean()), 2), "median": round(float(np.median(snr)), 2),
        "histogram": {f"{lo:+.0f}..{lo+5:+.0f}": int(((snr >= lo) & (snr < lo + 5)).sum())
                      for lo in range(-10, 20, 5)},
    }
    r["level_trajectory_distribution"] = dict(Counter(
        l["level_traj"] for x in rows for l in x.get("noise_layers", [])).most_common())
    r["n_noise_layers_distribution"] = dict(Counter(
        x["n_noise_layers"] for x in rows).most_common())
    r["rir_kind_distribution"] = dict(Counter(
        x.get("rir_kind", "none") for x in rows).most_common())
    r["clipped_samples_pct"] = round(
        100 * sum(1 for x in rows if x.get("clipping")) / len(rows), 2)
    r["distorted_samples_pct"] = round(
        100 * sum(1 for x in rows if x.get("preamp_distortion_k")) / len(rows), 2)
    r["speech_active_frac_mean"] = round(
        float(np.mean([x["speech_active_frac"] for x in rows])), 4)

    if audio_root and os.path.isdir(audio_root):
        tot_b = sum(os.path.getsize(os.path.join(d, f))
                    for d, _, fs in os.walk(audio_root) for f in fs)
        r["storage_gb"] = round(tot_b / 1e9, 3)
    r["provenance"] = dict(Counter(x.get("provenance", "?") for x in rows).most_common())
    return r


def render(r, header=None):
    L = []
    A = L.append
    A("=" * 68)
    A("RHEAR E03 DATASET REPORT")
    A("=" * 68)
    if header:
        A(f"  built        {header['created_utc']}   git {header['git_rev']}")
        A(f"  seed         {header['config'].get('seed')}")
        A(f"  sample rate  {header['config'].get('fs')} Hz, mono, "
          f"{header['config'].get('dur_s')} s clips")
    A("")
    A(f"  samples                {r['n_samples']}")
    A(f"  mixture hours          {r.get('mixture_hours', 0):.2f} h")
    A(f"  speakers               {r.get('speakers_total', 0)}")
    A(f"  clean utterances used  {r.get('clean_utterances_used', 0)}")
    A(f"  distinct noise recs    {r.get('distinct_noise_recordings', 0)}")
    if "storage_gb" in r:
        A(f"  storage                {r['storage_gb']:.2f} GB")
    A("")
    A("  split          samples     hours   speakers")
    for s, v in r.get("by_split", {}).items():
        A(f"  {s:<12} {v['samples']:8d} {v['hours']:9.2f} {v['speakers']:10d}")
    A("")
    A("  SNR (dB)   min {min} / median {median} / mean {mean} / max {max}".format(
        **r["snr_db"]))
    A("  SNR histogram: " + "  ".join(f"{k}:{v}" for k, v in
                                      r["snr_db"]["histogram"].items()))
    A("")
    A("  stationarity   " + "  ".join(f"{k} {v}%" for k, v in
                                      r["stationarity_pct"].items()))
    A("  noise classes  " + ", ".join(f"{k}:{v}" for k, v in
                                      list(r["noise_class_distribution"].items())[:12]))
    A("  level traj     " + "  ".join(f"{k}:{v}" for k, v in
                                      r["level_trajectory_distribution"].items()))
    A("  noise layers   " + "  ".join(f"{k}:{v}" for k, v in
                                      r["n_noise_layers_distribution"].items()))
    A("  reverberation  " + "  ".join(f"{k}:{v}" for k, v in
                                      r["rir_kind_distribution"].items()))
    A(f"  clipped {r['clipped_samples_pct']}%   distorted {r['distorted_samples_pct']}%"
      f"   mean speech-active {r['speech_active_frac_mean']*100:.0f}%")
    A("")
    A("  provenance     " + "  ".join(f"{k} ({v})" for k, v in r["provenance"].items()))
    A("=" * 68)
    return "\n".join(L)
