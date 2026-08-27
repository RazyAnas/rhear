#!/usr/bin/env python3
"""RHEAR Digital Twin -- persistent live visualisation.

    python3 rhear_twin.py                 # simulation source (default)
    python3 rhear_twin.py --source hw --port 8765

The dashboard is source-agnostic: simulation, laptop rig, dev board and headset
all publish the same TelemetryFrame, so the UI never changes as the programme
moves from twin to hardware.
"""
import argparse
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhear.telemetry.server import serve
from rhear.telemetry.bus import BUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim", choices=["sim", "hw"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--publish-hz", type=float, default=20.0)
    a = ap.parse_args()

    if a.source == "sim":
        from rhear.telemetry.sources.simulation import SimulationSource
        src = SimulationSource(publish_hz=a.publish_hz)
    else:
        from rhear.telemetry.sources.hardware import HardwareSource
        src = HardwareSource(publish_hz=a.publish_hz)

    serve(a.port)
    src.start()
    print(f"\n  RHEAR Digital Twin  ->  http://127.0.0.1:{a.port}")
    print(f"  source: {src.kind} ({src.name})   ctrl-C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        src.stop()


if __name__ == "__main__":
    main()
