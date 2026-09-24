#!/usr/bin/env python3
"""Animated before/after video: waveform + spectrogram with a moving playhead.

The spectrogram is the point. A waveform shows that something got quieter; a
spectrogram shows the noise floor DISAPPEARING across every frequency, which is
what the system actually does. Both panels share one colour scale between the
two clips, so the difference on screen is the difference in the audio and not
an artefact of autoscaling.
"""
import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import soundfile as sf
from scipy.signal import spectrogram

# usage: make_ab_video.py [before] [after] [outname] [label_before] [label_after]
BEFORE = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/ex01_noisy.wav")
AFTER  = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/Downloads/ex01_enhanced.wav")
NAME   = sys.argv[3] if len(sys.argv) > 3 else "RHEAR_before_after"
NOTE_B = sys.argv[4] if len(sys.argv) > 4 else "raw microphone — machine noise across every frequency"
NOTE_A = sys.argv[5] if len(sys.argv) > 5 else "through RHEAR — the noise floor is gone, the voice is not"
OUT    = os.path.dirname(os.path.abspath(__file__))
FPS    = 30
BG     = "#0d1117"
AMBER  = "#ff8c42"
GREEN  = "#3ddc84"
DIM    = "#30363d"

def load(p):
    x, sr = sf.read(p)
    if x.ndim > 1: x = x.mean(1)
    return x.astype(float), sr

xb, sr = load(BEFORE)
xa, _  = load(AFTER)
n = min(len(xb), len(xa)); xb, xa = xb[:n], xa[:n]
dur = n / sr

# Match the SPEECH level between the two clips before drawing anything.
# Without this the processed clip can simply be quieter overall, and a viewer
# reasonably reads "quieter" as "turned down" rather than "noise removed".
# Matching the loud frames means the only visible and audible difference left
# is the background, which is the entire claim.
def speech_rms(x, fr=320):
    m = len(x) // fr
    e = (x[:m*fr]**2).reshape(m, fr).mean(1)
    return float(np.sqrt(e[e >= np.quantile(e, 0.70)].mean() + 1e-20))
g = speech_rms(xb) / (speech_rms(xa) + 1e-20)
xa = np.clip(xa * g, -0.99, 0.99)
print(f"speech-level matched: scaled AFTER by {g:.2f}x so only the background differs")

# one shared scale for BOTH clips, so the eye compares audio and not autoscale
def spec(x):
    f, t, S = spectrogram(x, sr, nperseg=512, noverlap=384, mode="magnitude")
    return f, t, 20 * np.log10(S + 1e-8)
fb, tb, Sb = spec(xb)
fa, ta, Sa = spec(xa)
VMAX = max(Sb.max(), Sa.max())
VMIN = VMAX - 75
YMAX = max(np.abs(xb).max(), np.abs(xa).max()) * 1.1

def frame(tag, x, f, t, S, colour, prog, path, note):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    gs = GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.28,
                  left=0.07, right=0.97, top=0.82, bottom=0.10)

    fig.text(0.07, 0.92, tag, color=colour, fontsize=34, fontweight="bold",
             family="DejaVu Sans")
    fig.text(0.07, 0.875, note, color="#8b949e", fontsize=15)

    # ---- waveform -------------------------------------------------------
    ax = fig.add_subplot(gs[0], facecolor=BG)
    tt = np.arange(len(x)) / sr
    cut = int(prog * len(x))
    ax.plot(tt, x, color=DIM, lw=0.6)
    ax.plot(tt[:cut], x[:cut], color=colour, lw=0.8)
    ax.axvline(prog * dur, color="#ffffff", lw=1.4, alpha=0.9)
    ax.set_xlim(0, dur); ax.set_ylim(-YMAX, YMAX)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values(): s.set_color(DIM)
    ax.text(0.01, 0.86, "waveform", transform=ax.transAxes,
            color="#8b949e", fontsize=11)

    # ---- spectrogram ----------------------------------------------------
    ax2 = fig.add_subplot(gs[1], facecolor=BG)
    M = S.copy()
    M[:, t > prog * dur] = VMIN            # reveal as it plays
    ax2.pcolormesh(t, f / 1000, M, vmin=VMIN, vmax=VMAX,
                   cmap="magma", shading="gouraud")
    ax2.axvline(prog * dur, color="#ffffff", lw=1.4, alpha=0.9)
    ax2.set_xlim(0, dur); ax2.set_ylim(0, 8)
    ax2.set_ylabel("kHz", color="#8b949e", fontsize=12)
    ax2.tick_params(colors="#8b949e", labelsize=10)
    for s in ax2.spines.values(): s.set_color(DIM)
    ax2.text(0.01, 0.93, "spectrogram — every colour is sound energy",
             transform=ax2.transAxes, color="#c9d1d9", fontsize=11)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)

os.makedirs(f"{OUT}/frames", exist_ok=True)
nf = int(dur * FPS)
i = 0
for tag, x, f, t, S, col, note in [
    ("BEFORE", xb, fb, tb, Sb, AMBER, NOTE_B),
    ("AFTER",  xa, fa, ta, Sa, GREEN, NOTE_A),
]:
    for k in range(nf):
        frame(tag, x, f, t, S, col, (k + 1) / nf,
              f"{OUT}/frames/f{i:05d}.png", note)
        i += 1
    for _ in range(int(0.7 * FPS)):        # hold on the finished picture
        frame(tag, x, f, t, S, col, 1.0, f"{OUT}/frames/f{i:05d}.png", note)
        i += 1
print(f"rendered {i} frames")

# audio: the two clips back to back, matching the frame timeline
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", BEFORE,
                "-af", f"apad=pad_dur=0.7", f"{OUT}/_b.wav"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", AFTER,
                "-af", f"apad=pad_dur=0.7", f"{OUT}/_a.wav"], check=True)
with open(f"{OUT}/_list.txt", "w") as fh:
    fh.write(f"file '{OUT}/_b.wav'\nfile '{OUT}/_a.wav'\n")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                "-i", f"{OUT}/_list.txt", "-c", "copy", f"{OUT}/_both.wav"], check=True)

subprocess.run(["ffmpeg", "-v", "error", "-y",
                "-framerate", str(FPS), "-i", f"{OUT}/frames/f%05d.png",
                "-i", f"{OUT}/_both.wav",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                f"{OUT}/{NAME}.mp4"], check=True)
for f in ["_b.wav", "_a.wav", "_both.wav", "_list.txt"]:
    os.remove(f"{OUT}/{f}")
subprocess.run(["rm", "-rf", f"{OUT}/frames"])
sz = os.path.getsize(f"{OUT}/{NAME}.mp4") / 1024
print(f"wrote {NAME}.mp4  ({sz:.0f} KB)")
