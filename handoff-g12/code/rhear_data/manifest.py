"""Reproducible manifests.

Every generated sample records exactly how it was made: which real recordings it
came from, every transform and its parameters, and the RNG seed that produced it.
Re-running the build with the same config must reproduce the dataset bit for bit.
"""
import json
import os
import platform
import subprocess
import sys
import time


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_header(config, sources):
    import numpy, scipy
    return {
        "rhear_dataset_version": "0.1.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "config": config,
        "sources": sources,
        "provenance_legend": {
            "real recording": "unmodified audio from a public corpus",
            "synthetically mixed from real recordings":
                "real clean speech + real noise, combined and processed by this pipeline",
            "synthetically generated": "audio produced by a signal generator (NOT used here)",
        },
    }


class ManifestWriter:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "w")
        self.f.write(json.dumps({"_header": header}) + "\n")
        self.n = 0

    def add(self, rec):
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self):
        self.f.close()
        return self.n


def read_manifest(path):
    header, rows = None, []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_header" in d:
                header = d["_header"]
            else:
                rows.append(d)
    return header, rows
