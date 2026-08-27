"""Live hardware source.

Not yet connected to anything -- no hardware has been bought. It exists now so
the contract is fixed before the board arrives: when the MCU starts emitting
frames, ONLY this file changes. The dashboard, the schema and the bus are
already correct.

Wire protocol (planned): the MCU emits newline-delimited JSON over USB CDC at the
telemetry rate, containing the same field names as TelemetryFrame. Raw audio taps
are decimated on the device to keep the link under ~200 kB/s.

Until then this source publishes NOTHING rather than inventing values. The
dashboard shows "waiting for device", which is the honest display.
"""
import json
import time
from ..schema import TelemetryFrame, Block
from .base import TelemetrySource


class HardwareSource(TelemetrySource):
    kind = "headset"
    name = "waiting for device"

    def __init__(self, bus=None, publish_hz=20.0, port=None, baud=921600):
        super().__init__(bus=bus or __import__(
            "rhear.telemetry.bus", fromlist=["BUS"]).BUS, publish_hz=publish_hz)
        self.port = port
        self.baud = baud

    def _open(self):
        if self.port is None:
            return None
        try:
            import serial                       # pyserial, only needed on hardware
            return serial.Serial(self.port, self.baud, timeout=0.5)
        except Exception as ex:
            print(f"[hw] cannot open {self.port}: {ex}")
            return None

    def run(self):
        ser = self._open()
        t0 = time.time()
        while not self._stop.is_set():
            if ser is None:
                # Publish a heartbeat with every block idle and NO scalars, so the
                # dashboard renders "--" everywhere instead of stale or fake data.
                self.seq += 1
                self.bus.publish(TelemetryFrame(
                    t=time.time() - t0, seq=self.seq, source=self.kind,
                    experiment="no device connected", mode="NORMAL",
                    blocks={k: Block("idle", detail="no device")
                            for k in ("mic_ref_l", "mic_ref_r", "mic_err",
                                      "mic_boom", "afe", "codec", "l0", "l1",
                                      "l2", "driver", "ear", "radio")},
                    notes="HardwareSource: pass --port to connect a device"))
                time.sleep(1.0)
                continue
            line = ser.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.seq += 1
            d.setdefault("source", self.kind)
            d.setdefault("seq", self.seq)
            self.bus.publish(TelemetryFrame(**d))
