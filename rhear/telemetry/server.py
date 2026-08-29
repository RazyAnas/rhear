"""Zero-dependency telemetry server: static UI + Server-Sent Events.

SSE over stdlib http.server is chosen deliberately. Telemetry is one-way and
high-rate; EventSource reconnects on its own; and there is no framework to port
when this has to run on a dev board next to the headset.
"""
import json
import os
import threading
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs
from .bus import BUS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UI_DIR = os.path.join(ROOT, "ui")
# A/B demo assets produced by E03/evaluate_interim.py. Absent until it has run,
# in which case the panel simply does not appear -- no placeholder audio.
AB_DIR = os.path.join(ROOT, "E03", "runs", "demo_ab")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            p = os.path.join(UI_DIR, "index.html")
            with open(p, "rb") as f:
                return self._send(200, "text/html; charset=utf-8", f.read())
        if u.path == "/api/meta":
            from .schema import (SCHEMA_VERSION, FLOW_NODES, FLOW_EDGES,
                                 COEFFICIENT_EDGES)
            meta = {
                "schema": SCHEMA_VERSION,
                "nodes": [{"id": i, "label": l, "x": x, "y": y}
                          for i, l, x, y in FLOW_NODES],
                "edges": [{"src": a, "dst": b,
                           "coeff": (a, b) in COEFFICIENT_EDGES}
                          for a, b in FLOW_EDGES],
                "producer": BUS.producer_info,
            }
            return self._send(200, "application/json", json.dumps(meta))
        if u.path == "/api/ab":
            f = os.path.join(AB_DIR, "summary.json")
            if not os.path.exists(f):
                return self._send(404, "application/json", "{}")
            with open(f, "rb") as fh:
                return self._send(200, "application/json", fh.read())
        if u.path.startswith("/ab/"):
            name = os.path.basename(u.path[4:])
            f = os.path.join(AB_DIR, name)
            if not (name.endswith(".wav") and os.path.exists(f)):
                return self._send(404, "text/plain", "not found")
            with open(f, "rb") as fh:
                return self._send(200, "audio/wav", fh.read())
        if u.path == "/api/history":
            n = int(parse_qs(u.query).get("n", ["600"])[0])
            return self._send(200, "application/json",
                              json.dumps(BUS.history(n)))
        if u.path == "/api/stream":
            return self._stream()
        return self._send(404, "text/plain", "not found")

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = BUS.subscribe()
        try:
            while True:
                try:
                    frame = q.get(timeout=5.0)
                    payload = json.dumps(frame, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                except Exception as ex:
                    if isinstance(ex, (BrokenPipeError, ConnectionResetError)):
                        raise
                    self.wfile.write(b": keepalive\n\n")   # queue timeout
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            BUS.unsubscribe(q)


class ThreadedHTTP(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port=8765, block=False):
    srv = ThreadedHTTP(("127.0.0.1", port), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    if block:
        t.join()
    return srv
