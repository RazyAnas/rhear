#!/bin/bash
# Talk to the already-flashed board. Nothing is compiled or uploaded here.
#   ./run.sh          -> A/B test: records 5 s, plays RAW then GATED
#   ./run.sh live     -> continuous, mic -> suppressor -> gate -> speaker
#   ./run.sh meter    -> input level + mic format check
PORT=${PORT:-/dev/cu.usbmodem11401}
CMD=${1:-ab}
/opt/anaconda3/bin/python - "$PORT" "$CMD" <<'PY'
import serial, sys, time
port, cmd = sys.argv[1], sys.argv[2]
key = {"ab":"A", "live":"L", "meter":"M", "bypass":"B"}.get(cmd, "A")
s = serial.Serial(port, 2000000, timeout=1)
time.sleep(0.4); s.reset_input_buffer()
if key == "A":
    print("\n>>> Press ENTER, then TALK for 3 seconds, then GO QUIET. <<<")
    input()
s.write(key.encode())
print("--- board output (ctrl-C to stop) ---")
try:
    t0 = time.time()
    while True:
        l = s.readline().decode("utf-8", "replace").rstrip()
        if l: print(l); t0 = time.time()
        if l.startswith("done"): break
        if key != "L" and time.time() - t0 > 25: break
except KeyboardInterrupt:
    s.write(b"x")
s.close()
PY
