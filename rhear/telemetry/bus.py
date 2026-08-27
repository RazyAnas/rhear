"""Telemetry bus: producers publish frames, subscribers stream them.

Keeps a ring buffer so the UI can scrub back through history without the
producer having to store anything.
"""
import json
import math
import threading
import collections
from typing import Optional
from .schema import TelemetryFrame


def jsonable(o):
    """Coerce numpy scalars/arrays and non-finite floats to plain JSON.

    Done once at publish time so the ring buffer holds pure Python. Producers
    (simulation today, MCU tomorrow) therefore cannot break the wire format by
    leaking a numpy type -- which is exactly what happened on the first run:
    /api/meta worked, /api/history and the SSE stream silently returned nothing.
    """
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, str, type(None))):
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "item"):                       # numpy scalar
        try:
            v = o.item()
            return jsonable(v)
        except Exception:
            return None
    if hasattr(o, "tolist"):                     # numpy array
        return jsonable(o.tolist())
    return o


class TelemetryBus:
    def __init__(self, history=1200, max_queue=8):
        self._hist = collections.deque(maxlen=history)
        self._subs = []
        self._lock = threading.Lock()
        self._max_queue = max_queue
        self.seq = 0
        self.producer_info = {}

    def publish(self, frame: TelemetryFrame):
        d = jsonable(frame.to_dict())
        with self._lock:
            self._hist.append(d)
            dead = []
            for q in self._subs:
                try:
                    if q.qsize() < self._max_queue:
                        q.put_nowait(d)
                    # if a subscriber is behind, drop the frame for that
                    # subscriber rather than stalling the pipeline
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subs.remove(q)

    def subscribe(self):
        import queue
        q = queue.Queue(maxsize=self._max_queue * 4)
        with self._lock:
            self._subs.append(q)
            # prime with the newest frame so a fresh tab paints immediately
            if self._hist:
                q.put_nowait(self._hist[-1])
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def history(self, n: Optional[int] = None):
        with self._lock:
            h = list(self._hist)
        return h[-n:] if n else h

    def latest(self):
        with self._lock:
            return self._hist[-1] if self._hist else None


BUS = TelemetryBus()
