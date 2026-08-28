#!/usr/bin/env bash
# RHEAR — one command to start the demo. No internet needed.
set -e
cd "$(dirname "$0")"
PORT=${PORT:-8765}

echo
echo "  ┌────────────────────────────────────────────────────────┐"
echo "  │  RHEAR — AI-enabled adaptive noise cancellation        │"
echo "  │  SIH PS 26052 · DRDO · Smart Vehicles                  │"
echo "  └────────────────────────────────────────────────────────┘"
echo

pkill -f rhear_twin.py 2>/dev/null || true
sleep 1
nohup python3 -u rhear_twin.py --port "$PORT" > /tmp/rhear_twin.log 2>&1 &
echo "  starting live twin ..."
for i in $(seq 1 60); do
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/api/history?n=1" | grep -q '"seq"'; then
    echo "  LIVE  ->  http://127.0.0.1:$PORT"
    echo
    curl -s "http://127.0.0.1:$PORT/api/history?n=1" | python3 -c "
import json,sys
d=json.load(sys.stdin)[-1]; s=d['scalars']
print(f\"  scene {d['notes']}\")
print(f\"  attenuation  L {s['atten_L']['value']:+.1f} dB   R {s['atten_R']['value']:+.1f} dB\")
print(f\"  real-time factor {s['rtf']['value']:.2f}  (must be < 1)\")
"
    echo
    echo "  open the URL in a browser. ctrl-C here to stop."
    exit 0
  fi
  sleep 1
done
echo "  FAILED to start. Log:"; tail -20 /tmp/rhear_twin.log; exit 1
