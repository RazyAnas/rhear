"""Base class for telemetry producers.

A producer runs a pipeline and publishes TelemetryFrames. Simulation, laptop rig,
dev board and headset all subclass this, so the dashboard cannot tell them apart
except by the `source` field it displays.
"""
import threading
import time
from ..bus import BUS


class TelemetrySource:
    kind = "sim"
    name = "unnamed"

    def __init__(self, bus=BUS, publish_hz=20.0):
        self.bus = bus
        self.publish_hz = publish_hz
        self._stop = threading.Event()
        self._thread = None
        self.seq = 0
        self.t0 = None

    def start(self):
        self.t0 = time.time()
        self.bus.producer_info = {"kind": self.kind, "name": self.name}
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_guarded(self):
        try:
            self.run()
        except Exception:
            import traceback
            traceback.print_exc()

    def run(self):
        raise NotImplementedError
