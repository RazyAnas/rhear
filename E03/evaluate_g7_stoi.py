#!/usr/bin/env python3
"""Evaluate G7 on the held-out test split using STOI + SI-SDR only."""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "model"))

from gtcrn_lite import (
    GTCRNLite,
    N_FFT,
    FS,
    count_params,
    count_macs_per_frame,
    HOP,
)

from train_interim import enhance, si_sdr, Pairs, DEV
from rhear_data.manifest import read_manifest
from pystoi import stoi as stoi_fn


def score(clean, noisy, enh, fs=FS):
    out = {}

    # STOI
    for tag, sig in (
        ("noisy", noisy),
        ("enhanced", enh),
    ):
        try:
            out[f"stoi_{tag}"] = float(
                stoi_fn(clean, sig, fs, extended=False)
            )
        except Exception:
            out[f"stoi_{tag}"] = float("nan")

    # SI-SDR
    c = torch.from_numpy(clean)[None]
    out["sisdr_noisy"] = float(
        si_sdr(torch.from_numpy(noisy)[None], c)
    )

    out["sisdr_enhanced"] = float(
        si_sdr(torch.from_numpy(enh)[None], c)
    )

    return out


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--ckpt",
        default="runs/interim/best.pt"
    )
    ap.add_argument(
        "--out",
        default="runs/interim/eval_stoi"
    )
    ap.add_argument(
        "--n-examples",
        type=int,
        default=8
    )

    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # Read manifest
    hdr, rows = read_manifest(
        os.path.join(a.data, "manifest.jsonl")
    )

    # IMPORTANT:
    # Evaluate only the held-out TEST split.
    ds = Pairs(
        a.data,
        "test",
        rows,
        cache=False
    )

    # Load G7 model
    model = GTCRNLite(
    hop=256,
    ch=(32, 48, 48, 64),
    fullband=True,
    phase=False,
    df=True,
).to(DEV)

    model.load_state_dict(
        torch.load(
            a.ckpt,
            map_location=DEV
        )
    )

    model.eval()

    win = torch.hann_window(
        N_FFT,
        device=DEV
    )

    print(
        f"  {len(ds)} test clips"
        f"  | ckpt {a.ckpt}"
        f"  | device {DEV}"
    )

    res = []
    per_class = {}

    # Evaluate every test sample
    for i in range(len(ds)):

        x, y = ds[i]

        with torch.no_grad():

            _, est, _, _ = enhance(
                model,
                x[None].to(DEV),
                win
            )

        enhanced = (
            est[0]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        clean = (
            y.numpy()
            .astype(np.float64)
        )

        noisy = (
            x.numpy()
            .astype(np.float64)
        )

        s = score(
            clean,
            noisy,
            enhanced
        )

        r = ds.rows[i]

        s.update(
            id=r["id"],
            snr_db=r["snr_db"],
            stationarity=r["mixture_stationarity"],
            classes=[
                l["cls"]
                for l in r["noise_layers"]
            ],
        )

        res.append(s)

        per_class.setdefault(
            r["mixture_stationarity"],
            []
        ).append(s)

        if i % 50 == 0:
            print(
                f"    {i}/{len(ds)}"
            )

    # Aggregate metric
    def agg(rs, key):

        values = [
            r[key]
            for r in rs
            if np.isfinite(r[key])
        ]

        return (
            float(np.mean(values))
            if values
            else float("nan")
        )

    # Overall summary
    summary = {

        "WARNING":
            "REAL SPEECH + REAL NOISE. "
            "STOI/SI-SDR evaluation only.",

        "checkpoint":
            a.ckpt,

        "n_test":
            len(res),

        "model": {
            "params":
                count_params(model),

            "mmacs":
                count_macs_per_frame(model)[0]
                * (FS / HOP)
                / 1e6,

            "latency_ms":
                8.0,
        },

        "overall": {

            "stoi_noisy":
                round(
                    agg(res, "stoi_noisy"),
                    4
                ),

            "stoi_enhanced":
                round(
                    agg(res, "stoi_enhanced"),
                    4
                ),

            "sisdr_noisy":
                round(
                    agg(res, "sisdr_noisy"),
                    4
                ),

            "sisdr_enhanced":
                round(
                    agg(res, "sisdr_enhanced"),
                    4
                ),
        },

        "by_stationarity": {

            c: {

                "stoi_noisy":
                    round(
                        agg(v, "stoi_noisy"),
                        4
                    ),

                "stoi_enhanced":
                    round(
                        agg(v, "stoi_enhanced"),
                        4
                    ),

                "sisdr_noisy":
                    round(
                        agg(v, "sisdr_noisy"),
                        4
                    ),

                "sisdr_enhanced":
                    round(
                        agg(v, "sisdr_enhanced"),
                        4
                    ),

            }

            for c, v in per_class.items()
        },
    }

    # Metric improvements
    o = summary["overall"]

    summary["deltas"] = {

        "d_stoi":
            round(
                o["stoi_enhanced"]
                - o["stoi_noisy"],
                4
            ),

        "d_sisdr":
            round(
                o["sisdr_enhanced"]
                - o["sisdr_noisy"],
                4
            ),
    }

    # Save detailed results
    with open(
        os.path.join(
            a.out,
            "summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    with open(
        os.path.join(
            a.out,
            "results.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            res,
            f,
            indent=2
        )

    print()
    print("=" * 60)
    print("G7 EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"STOI  noisy:    "
        f"{o['stoi_noisy']:.4f}"
    )

    print(
        f"STOI  enhanced: "
        f"{o['stoi_enhanced']:.4f}"
    )

    print(
        f"STOI  delta:    "
        f"{summary['deltas']['d_stoi']:+.4f}"
    )

    print()

    print(
        f"SI-SDR noisy:    "
        f"{o['sisdr_noisy']:.4f} dB"
    )

    print(
        f"SI-SDR enhanced: "
        f"{o['sisdr_enhanced']:.4f} dB"
    )

    print(
        f"SI-SDR delta:    "
        f"{summary['deltas']['d_sisdr']:+.4f} dB"
    )

    print()

    print(
        f"Results saved to: "
        f"{a.out}"
    )

    print("=" * 60)