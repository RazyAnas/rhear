#!/usr/bin/env python3
"""RHEAR Phase 0 -- Milestone 5.1b: on-cup coherence and the cancellation ceiling.

Two subcommands:

    capture   read the dual-ADC stream off the ESP32-S3 and save it
    analyse   turn a saved capture into the number and the plot

They are separate on purpose: the analysis can be re-run on the saved file
during the demo with no hardware attached, and the capture is never lost to a
plotting bug.

    /opt/anaconda3/bin/python demo/coherence.py capture --seconds 20 --out demo/captures/cup
    /opt/anaconda3/bin/python demo/coherence.py analyse demo/captures/cup.npz

What it measures
----------------
gamma^2(f), the magnitude-squared coherence between the reference microphone
(outside the cup) and the error microphone (at the ear). Feed-forward ANC cannot
beat

    NR(f) <= -10 * log10(1 - gamma^2(f))

no matter what the algorithm is. That is the ceiling this measurement finds for
*this* headset with *this* microphone placement.

Report the CUP number, not a free-air number. Free air always looks better and
it is not the system.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


FS_DEFAULT = 8000
TAG_MASK = 0xF000
TAG_REF = 0x0000
TAG_ERR = 0x1000
ADC_MAX = 4095

BANDS = [(100, 500), (500, 1000), (1000, 2000), (2000, 4000)]


# ----------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------

def find_port() -> str:
    import serial.tools.list_ports as lp

    cands = [
        p.device
        for p in lp.comports()
        if "usbmodem" in p.device or "usbserial" in p.device or "wchusb" in p.device
    ]
    if not cands:
        sys.exit(
            "No USB serial port found.\n"
            "Plug the ESP32-S3 in, then re-run. On macOS the port looks like "
            "/dev/cu.usbmodem1101.\n"
            "Bluetooth audio devices also show up as /dev/cu.* -- they are not it."
        )
    if len(cands) > 1:
        print(f"note: several ports found {cands}, using {cands[0]}", file=sys.stderr)
    return cands[0]


def cmd_capture(a: argparse.Namespace) -> None:
    import serial

    port = a.port or find_port()
    out = Path(a.out).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"port      {port}")
    ser = serial.Serial(port, a.baud, timeout=2)
    time.sleep(0.3)
    ser.reset_input_buffer()

    # The sketch prints one ASCII banner line, then never sends ASCII again.
    banner = ser.readline().decode("ascii", "replace").strip()
    if "RHEAR_DUAL_ADC" not in banner:
        print(
            f"warning: expected the RHEAR_DUAL_ADC banner, got {banner!r}.\n"
            "         Reset the board and re-run if the capture looks wrong.",
            file=sys.stderr,
        )
    else:
        print(f"banner    {banner}")
        for tok in banner.split():
            if tok.startswith("fs="):
                fw_fs = int(tok[3:])
                if fw_fs != a.fs:
                    sys.exit(
                        f"Sample-rate mismatch: firmware says {fw_fs} Hz, this "
                        f"run assumes {a.fs} Hz. Coherence computed against the "
                        "wrong fs is meaningless -- pass --fs {fw_fs}."
                    )

    need = a.fs * 4 * a.seconds
    print(f"recording {a.seconds}s  ({need} bytes)")
    print("play broadband noise about a metre away NOW")

    buf = bytearray()
    t0 = time.time()
    while len(buf) < need:
        chunk = ser.read(min(4096, need - len(buf)))
        if not chunk:
            sys.exit(
                f"Serial went quiet after {len(buf)} of {need} bytes. "
                "Check the board is still running the sketch."
            )
        buf += chunk
        done = len(buf) / need
        print(f"\r  {done * 100:5.1f}%  {time.time() - t0:5.1f}s", end="", flush=True)
    print()
    ser.close()

    raw = np.frombuffer(bytes(buf), dtype="<u2")

    # Lock onto the stream: the first sample tagged REF whose successor is ERR.
    start = None
    for i in range(min(64, len(raw) - 1)):
        if (raw[i] & TAG_MASK) == TAG_REF and (raw[i + 1] & TAG_MASK) == TAG_ERR:
            start = i
            break
    if start is None:
        sys.exit("Could not find a REF/ERR sample pair -- is the board running this sketch?")
    raw = raw[start:]
    raw = raw[: (len(raw) // 2) * 2]

    tags_ok = int(
        np.sum((raw[0::2] & TAG_MASK) != TAG_REF) + np.sum((raw[1::2] & TAG_MASK) != TAG_ERR)
    )
    if tags_ok:
        print(
            f"warning: {tags_ok} samples had the wrong channel tag -- bytes were "
            "dropped on the USB link. Lower --baud or shorten --seconds.",
            file=sys.stderr,
        )

    ref = (raw[0::2] & ~TAG_MASK).astype(np.float64)
    err = (raw[1::2] & ~TAG_MASK).astype(np.float64)

    np.savez_compressed(out, ref=ref, err=err, fs=a.fs, tag_errors=tags_ok)
    print(f"saved     {out}  ({len(ref)} samples, {len(ref) / a.fs:.1f}s)")
    print()
    _analyse(ref, err, a.fs, out.with_suffix(""), write_wav=a.wav)


# ----------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------

def _health(name: str, x: np.ndarray) -> None:
    """Flag the two failure modes that fake a low coherence result."""
    rail_lo = float(np.mean(x <= 1))
    rail_hi = float(np.mean(x >= ADC_MAX - 1))
    rms = float(np.std(x))
    print(f"  {name:9s} mean {x.mean():7.1f}   rms {rms:7.1f} counts", end="")
    if rail_lo + rail_hi > 0.001:
        print(f"   *** CLIPPING {100 * (rail_lo + rail_hi):.2f}% of samples ***")
    elif rms < 15:
        print("   *** NEARLY SILENT -- turn the gain up or the noise source louder ***")
    else:
        print()


def _analyse(ref, err, fs, stem: Path, write_wav: bool = True) -> None:
    from scipy.signal import coherence, welch
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("channel health (before DC removal):")
    _health("reference", ref)
    _health("error", err)

    ref = ref - ref.mean()
    err = err - err.mean()

    ratio = np.std(ref) / max(np.std(err), 1e-9)
    print(f"  gain ratio ref/err = {ratio:.2f}", end="")
    if not 0.5 <= ratio <= 2.0:
        print("   <-- trim the MAX4466 pots until this is near 1")
    else:
        print()
    print()

    nper = 1024
    f, cxy = coherence(ref, err, fs=fs, nperseg=nper)
    _, pref = welch(ref, fs=fs, nperseg=nper)

    bound = -10 * np.log10(np.clip(1 - cxy, 1e-6, None))

    print(f"coherence, {len(ref) / fs:.1f}s at {fs} Hz, nperseg={nper}")
    print("  band            gamma^2    max cancellation")
    rows = []
    for lo, hi in BANDS:
        s = (f >= lo) & (f < hi)
        # Power-weighted: bands where there is no noise energy carry no
        # information about the ceiling, and a flat mean lets them dominate.
        w = pref[s]
        c = float(np.sum(cxy[s] * w) / max(np.sum(w), 1e-30))
        b = -10 * np.log10(max(1 - c, 1e-6))
        rows.append((lo, hi, c, b))
        print(f"  {lo:5d}-{hi:5d} Hz   {c:6.3f}     {b:5.1f} dB")
    print()

    low = rows[0][3]
    if low >= 15:
        print("VERDICT  geometry is fine at low frequency; the design's claims hold.")
    elif low >= 10:
        print("VERDICT  workable, but the ceiling is tighter than the documents assume.")
    else:
        print(
            "VERDICT  ceiling is BELOW 10 dB at low frequency. That is a genuine\n"
            "         finding -- report it. The lever is moving the reference mic\n"
            "         closer to the cup: less decorrelation, less causality margin."
        )
    print()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.semilogx(f[1:], cxy[1:], lw=1.2)
    ax1.set_ylabel(r"coherence  $\gamma^2$")
    ax1.set_ylim(0, 1)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_title("RHEAR -- measured cancellation ceiling, 3M Peltor X3A, on-cup")

    ax2.semilogx(f[1:], bound[1:], lw=1.2, color="C3")
    ax2.set_ylabel("max cancellation (dB)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylim(0, max(25, float(np.percentile(bound[1:], 99)) + 3))
    ax2.grid(True, which="both", alpha=0.3)

    # Stagger the labels: adjacent bands can land at nearly the same dB and
    # the text would otherwise overlap.
    for i, (lo, hi, c, b) in enumerate(rows):
        ax2.hlines(b, lo, hi, color="k", lw=2.5)
        ax2.annotate(
            f"{b:.1f} dB", xy=(np.sqrt(lo * hi), b),
            xytext=(0, 7 if i % 2 == 0 else -15),
            textcoords="offset points", ha="center", fontsize=9,
        )

    ax1.set_xlim(50, fs / 2)

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(
        0.5, 0.012,
        r"$NR \leq -10\,\log_{10}(1-\gamma^2)$   —   no algorithm beats this",
        ha="center", va="bottom", fontsize=11,
    )
    png = stem.with_name(stem.name + "_coherence.png")
    fig.savefig(png, dpi=160)
    print(f"plot      {png}")

    if write_wav:
        import soundfile as sf

        def norm(x):
            return (x / max(np.abs(x).max(), 1e-9) * 0.9).astype(np.float32)

        wav = stem.with_name(stem.name + "_refA_errB.wav")
        sf.write(wav, np.stack([norm(ref), norm(err)], axis=1), fs)
        print(f"audio     {wav}  (ch1 = reference, ch2 = error)")


def cmd_analyse(a: argparse.Namespace) -> None:
    p = Path(a.npz)
    d = np.load(p)
    if int(d.get("tag_errors", 0)):
        print(f"note: this capture had {int(d['tag_errors'])} tag errors\n", file=sys.stderr)
    _analyse(d["ref"].astype(float), d["err"].astype(float), int(d["fs"]),
             p.with_suffix(""), write_wav=a.wav)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="record from the ESP32-S3")
    c.add_argument("--port", default=None, help="default: autodetect /dev/cu.usbmodem*")
    c.add_argument("--baud", type=int, default=921600)
    c.add_argument("--fs", type=int, default=FS_DEFAULT)
    c.add_argument("--seconds", type=int, default=20)
    c.add_argument("--out", default="demo/captures/cup")
    c.add_argument("--no-wav", dest="wav", action="store_false")
    c.set_defaults(func=cmd_capture, wav=True)

    n = sub.add_parser("analyse", help="re-run the numbers on a saved capture")
    n.add_argument("npz")
    n.add_argument("--no-wav", dest="wav", action="store_false")
    n.set_defaults(func=cmd_analyse, wav=True)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
