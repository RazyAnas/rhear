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
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rhear.telemetry.server import serve
from rhear.telemetry.bus import BUS


WATCH_DIRS = ("rhear",)
WATCH_EXT = (".py",)


def _snapshot():
    """mtimes of every source file the running pipeline depends on."""
    root = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for d in WATCH_DIRS:
        for dirpath, _, files in os.walk(os.path.join(root, d)):
            for f in files:
                if f.endswith(WATCH_EXT):
                    p = os.path.join(dirpath, f)
                    try:
                        out[p] = os.path.getmtime(p)
                    except OSError:
                        pass
    return out


def _watch_and_restart(interval=2.0):
    """Python imports a module once, so editing rhear/*.py does NOT affect a
    running twin -- it keeps executing the code it started with. This watcher
    makes 'the dashboard updates when you change the code' actually true by
    re-exec'ing the process when a source file changes.

    (The UI file is served from disk per request and already picks up edits on a
    browser reload; only the Python side needs this.)
    """
    base = _snapshot()
    while True:
        time.sleep(interval)
        cur = _snapshot()
        changed = [p for p, m in cur.items() if base.get(p) != m]
        if changed:
            names = ", ".join(os.path.basename(p) for p in sorted(changed)[:4])
            print(f"\n  [reload] {len(changed)} file(s) changed ({names}) - restarting\n",
                  flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sim", choices=["sim", "hw"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--publish-hz", type=float, default=20.0)
    ap.add_argument("--no-reload", action="store_true",
                    help="disable auto-restart on source changes")
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
    print(f"  source: {src.kind} ({src.name})")
    if not a.no_reload:
        threading.Thread(target=_watch_and_restart, daemon=True).start()
        print("  auto-reload: ON (edits to rhear/ restart the pipeline)")
    print("  ctrl-C to stop\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        src.stop()


if __name__ == "__main__":
    main()
