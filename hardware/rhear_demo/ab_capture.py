#!/usr/bin/env python3
"""Save the ESP32's A/B capture as two WAVs.

    python3 ab_capture.py /dev/cu.usbmodem11401
    (then press W in the Arduino serial monitor, or let this send it)

Writes ab_raw.wav and ab_enhanced.wav next to this script.
"""
import sys, serial, wave, time

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem11401"
ser = serial.Serial(port, 2000000, timeout=30)
time.sleep(2.0)
ser.reset_input_buffer()
ser.write(b"W")
print("sent W -- talk for 4 seconds")

n_raw = n_enh = None
while True:
    line = ser.readline().decode("utf-8", "replace").strip()
    if not line:
        continue
    print("  ", line)
    if line.startswith("ABDATA"):
        _, a, b = line.split()
        n_raw, n_enh = int(a), int(b)
        break

def grab(n):
    need = n * 2
    buf = b""
    while len(buf) < need:
        chunk = ser.read(need - len(buf))
        if not chunk:
            raise RuntimeError(f"short read: {len(buf)}/{need}")
        buf += chunk
    return buf

raw, enh = grab(n_raw), grab(n_enh)
for name, data in (("ab_raw.wav", raw), ("ab_enhanced.wav", enh)):
    with wave.open(name, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(data)
    print("wrote", name, len(data)//2, "samples")
ser.close()
