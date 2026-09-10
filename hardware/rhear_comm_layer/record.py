#!/usr/bin/env python3
"""Save the ESP32's 10 s recording as a WAV.

    python3 record.py /dev/cu.usbmodem1401

Sends 'R', waits for the REC10 marker, reads 10 s of int16 @16 kHz.
"""
import sys, time, wave, serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem1401"
FS, SECS = 16000, 10
ser = serial.Serial(port, 2000000, timeout=30)
time.sleep(2.0)
ser.reset_input_buffer()
ser.write(b"R")
print("sent R -- talk for 10 seconds")

while True:
    line = ser.readline().decode("utf-8", "replace").strip()
    if not line:
        continue
    print("  ", line)
    if line.startswith("REC10"):
        break

need = FS * SECS * 2
buf = b""
while len(buf) < need:
    chunk = ser.read(need - len(buf))
    if not chunk:
        raise RuntimeError(f"short read {len(buf)}/{need}")
    buf += chunk
    print(f"\r  {100*len(buf)//need:3d}%", end="", flush=True)
print()

with wave.open("mic_record.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(FS)
    w.writeframes(buf)
print("wrote mic_record.wav", len(buf)//2, "samples")
ser.close()
