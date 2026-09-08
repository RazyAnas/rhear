#!/usr/bin/env python3
"""RHEAR L1 communication layer -- host side.

  python3 ~/rhear_comms.py record 10      capture N seconds, enhance, save both
  python3 ~/rhear_comms.py tone           1 kHz on the speaker (amp/speaker check)
  python3 ~/rhear_comms.py passthru       mic -> speaker on the board, no host
  python3 ~/rhear_comms.py status         report the board's pin map

INPUT CONDITIONING -- both of these were found the hard way on a real capture:

 1. HIGH-PASS at 90 Hz. The INMP441's raw output is dominated by sub-20 Hz
    content: on the first real capture, 0-20 Hz held essentially ALL the energy
    and the 300-3400 Hz speech band sat 19 dB below it. The training pipeline
    models the mic as a 1st-order high-pass drawn from 40-140 Hz (median 87), so
    the model has never seen un-filtered DC and rumble.

 2. LEVEL NORMALISE to -18 dBFS. Training mixtures average -18.1 dBFS RMS; the
    raw capture was -31 to -39 dBFS after filtering, 13-21 dB quieter than
    anything the model was trained on. Left un-normalised the model treated the
    quiet half as noise-only and suppressed it by 51 dB -- it silenced real
    speech. With normalisation that gap falls to 6.5 dB.

Neither is a model fault; both are the conditioning any deployed front end owes
the network, and the ADAU1772 would do the first in hardware.
"""
import sys, time, wave, os
import numpy as np
import serial
from scipy import signal as sg

PORT = next((f"/dev/{d}" for d in sorted(os.listdir("/dev"))
             if d.startswith("cu.usbmodem")), None)
BAUD, FS, HP_HZ, TARGET_DBFS = 2000000, 16000, 90.0, -18.0

def open_board():
    if not PORT: sys.exit("No ESP32 found. Plug it into the Mac directly.")
    p = serial.Serial(PORT, BAUD, timeout=2); time.sleep(2); p.reset_input_buffer()
    return p

def condition(x, fs=FS):
    b, a = sg.butter(2, HP_HZ / (fs / 2), btype="high")
    y = sg.lfilter(b, a, x).astype(np.float32)
    y -= y.mean()
    rms = np.sqrt((y ** 2).mean()) + 1e-12
    return (y * 10 ** ((TARGET_DBFS - 20 * np.log10(rms)) / 20)).astype(np.float32)

def enhance_file(x, chunk_s=4.0, ov_s=0.5):
    """Run the model in TRAINING-LENGTH chunks with an overlap-add crossfade.

    Found on a real 10 s capture: in one pass the mask collapsed to ~0.002 after
    about 5 s while the speech-presence head still read 1.000 -- the two heads
    contradicted each other. Training clips are 4 s (250 frames); a 10 s input is
    626 frames, so the DPRNN's recurrent state runs 2.5x beyond anything it saw
    in training and drifts. Chunking at 4 s took the first-half/second-half level
    gap from 6.5 dB to 0.4 dB.

    A continuously streaming deployment has the same exposure and needs the same
    treatment -- periodic state reset, or training on longer sequences.
    """
    sys.path.insert(0, os.path.expanduser("~/PS#2/E03"))
    sys.path.insert(0, os.path.expanduser("~/PS#2/E03/model"))
    import torch
    from train_interim import enhance, DEV
    from gtcrn_lite import GTCRNLite, N_FFT
    ck = os.path.expanduser("~/PS#2/E03/runs/g12_noise232/best.pt")
    sd = torch.load(ck, map_location=DEV)
    m = GTCRNLite(ch=(32, 48, 48, 64), n_bands=sd["erb.M"].shape[0], phase=False,
                  fullband=True, df=False, hop=256).to(DEV)
    m.load_state_dict(sd); m.eval()
    win = torch.hann_window(N_FFT, device=DEV)

    def one(seg):
        with torch.no_grad():
            _, y, _, _ = enhance(m, torch.from_numpy(np.ascontiguousarray(seg)
                                                     ).to(DEV).unsqueeze(0), win)
        return y.squeeze(0).cpu().numpy()

    C, O = int(chunk_s * FS), int(ov_s * FS)
    if len(x) <= C:
        return one(x)
    out = np.zeros(len(x), dtype=np.float32)
    wsum = np.zeros(len(x), dtype=np.float32)
    i = 0
    while i < len(x):
        seg = x[i:i + C]
        if len(seg) < int(0.5 * FS):
            break
        y = one(seg)
        w = np.ones(len(y), dtype=np.float32)
        f = min(O, len(y) // 2)
        if f > 0:
            w[:f] = np.linspace(0, 1, f)
            w[-f:] = np.linspace(1, 0, f)
        out[i:i + len(y)] += y * w
        wsum[i:i + len(y)] += w
        i += C - O
    return out / np.maximum(wsum, 1e-6)

def save(path, x):
    x = np.clip(x / (np.abs(x).max() + 1e-9) * 0.9, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
        w.writeframes((x * 32767).astype("<i2").tobytes())

def cmd_record(secs):
    p = open_board()
    want = FS * secs * 2
    print(f"recording {secs}s from the INMP441 -- speak now")
    p.write(b"R"); buf = bytearray()
    while len(buf) < want:
        d = p.read(min(8192, want - len(buf)))
        if not d: break
        buf.extend(d)
        print(f"\r  {len(buf)*100//want}%", end="", flush=True)
    p.close(); print()
    raw = np.frombuffer(bytes(buf[:want]), dtype="<i2").astype(np.float32) / 32768.0
    save(os.path.expanduser("~/rhear_raw.wav"), raw)
    cond = condition(raw)
    out = enhance_file(cond)
    save(os.path.expanduser("~/rhear_enhanced.wav"), out)
    r = lambda v: 20*np.log10(np.sqrt((v**2).mean())+1e-12)
    print(f"  raw       {r(raw):6.1f} dBFS  -> ~/rhear_raw.wav")
    print(f"  condition {r(cond):6.1f} dBFS  (90 Hz HPF + level norm)")
    print(f"  enhanced  {r(out):6.1f} dBFS  -> ~/rhear_enhanced.wav")

def cmd_simple(ch, label, wait=8):
    p = open_board(); print(label); p.write(ch)
    t0 = time.time(); buf = b""
    while time.time() - t0 < wait:
        d = p.read(256)
        if d: buf += d
        if b"DONE" in buf or b"ERR" in buf or b"\n" in buf: break
    print("  board:", buf.decode("utf-8", "replace").strip() or "(no reply)")
    p.close()

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "record"
    if a == "record":   cmd_record(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif a == "tone":   cmd_simple(b"T", "1 kHz tone on the speaker -- listen")
    elif a == "status": cmd_simple(b"?", "board status:")
    elif a == "passthru":
        p = open_board(); print("mic -> speaker. ctrl-C to stop."); p.write(b"P")
        try:
            while True: time.sleep(0.5)
        except KeyboardInterrupt:
            p.write(b"X"); time.sleep(0.3); p.close(); print("\nstopped")
    else: sys.exit(__doc__)
