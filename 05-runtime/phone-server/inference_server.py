#!/usr/bin/env python3
"""
Integrated phone -> inference server (low-latency alternative to https_server.py).

Old path: phone POSTs a frame -> saved to disk -> a separate demo.py polls the
folder every 0.5s -> reads it back -> infers. Disk + polling sit in the hot path.

Here, each uploaded frame is decoded in memory and handed to a single background
worker that runs the refactored Pipeline on the LATEST frame only (stale frames
are dropped, so there's never a backlog and we always process the freshest one).
The current board state is returned on every upload and served at GET /state, so
the phone (or CORMAS) reads one endpoint instead of watching a directory.

  python inference_server.py [--port 8443] [--json-out <dir>] [--ws-url ...]
Open https://<LAPTOP-IP>:8443/camera_client.html on the phone.
"""
from __future__ import annotations
import argparse
import base64
import json
import re
import ssl
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for demo.py / json_bridge.py
from demo import Pipeline, resolve_weights_path, cell_counts  # noqa: E402
from json_bridge import build_state, write_state  # noqa: E402

_DATA_URL = re.compile(r"^data:image/\w+;base64,")


class Shared:
    """Thread-safe handoff: newest pending frame + latest computed state."""
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = None          # np.ndarray awaiting inference (latest wins)
        self.pending_id = 0
        self.processed_id = 0
        self.state = {}              # latest board-state dict

    def put_frame(self, img):
        with self.lock:
            self.pending = img
            self.pending_id += 1

    def take_frame(self):
        with self.lock:
            if self.pending_id <= self.processed_id:
                return None, None
            return self.pending, self.pending_id

    def set_state(self, state, fid):
        with self.lock:
            self.state = state
            self.processed_id = fid

    def get_state(self):
        with self.lock:
            return dict(self.state)


def process_latest(pipe, shared: Shared, session: str, json_out=None) -> bool:
    """One worker step: infer the freshest pending frame; update shared state. Returns True if it ran."""
    frame, fid = shared.take_frame()
    if frame is None:
        return False
    outcome = pipe.process(frame)
    counts = cell_counts(outcome.mapped, outcome.class_names) if outcome else {}
    state = build_state(session, f"frame_{fid:06d}", counts, pipe.rows, pipe.cols)
    shared.set_state(state, fid)
    if json_out:
        try:
            write_state(json_out, session, f"frame_{fid:06d}", counts, pipe.rows, pipe.cols)
        except Exception as e:
            print(f"[server] JSON write failed: {e}")
    if pipe.client and outcome:
        from demo import group_cells_by_class
        try:
            pipe.client.send_class_map(group_cells_by_class(outcome.mapped, outcome.class_names))
        except Exception as e:
            print(f"[server] WS send failed: {e}")
    return True


def worker(pipe, shared: Shared, session: str, json_out, stop: threading.Event):
    while not stop.is_set():
        if not process_latest(pipe, shared, session, json_out):
            time.sleep(0.01)


def make_handler(shared: Shared, page: Path):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code=200, body=b"", ctype="text/plain", cookie=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(200, json.dumps(shared.get_state()).encode(), "application/json")
            elif self.path in ("/", "/camera_client.html"):
                self._send(200, page.read_bytes(), "text/html")
            else:
                self._send(404)

        def do_POST(self):
            if self.path == "/start_session":
                sid = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self._send(200, b"OK", cookie=f"session_id={sid}; Path=/")
            elif self.path == "/upload_frame":
                n = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n).decode("utf-8", "ignore")
                raw = base64.b64decode(_DATA_URL.sub("", body))
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    shared.put_frame(img)
                self._send(200, json.dumps(shared.get_state()).encode(), "application/json")
            else:
                self._send(404)
    return Handler


def run(pipe, port, json_out, certfile, keyfile, page):
    shared = Shared()
    session = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stop = threading.Event()
    threading.Thread(target=worker, args=(pipe, shared, session, json_out, stop), daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", port), make_handler(shared, page))
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(f"[server] inference server on https://0.0.0.0:{port}  (session {session})")
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        pipe.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--ws-url", type=str, default=None)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.45)
    args = ap.parse_args()
    weights = resolve_weights_path(Path(args.weights) if args.weights else None)
    print(f"[server] weights: {weights}")
    pipe = Pipeline(weights, rows=args.rows, cols=args.cols, conf=args.conf,
                    json_out=None, ws_url=args.ws_url)
    run(pipe, args.port, args.json_out, HERE / "cert.pem", HERE / "key.pem", HERE / "camera_client.html")


if __name__ == "__main__":
    main()
