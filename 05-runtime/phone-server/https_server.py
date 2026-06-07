import http.server
import socketserver
import ssl
import os
import re
import base64
from datetime import datetime

PORT = 8443
DIRECTORY = "."

os.makedirs("frames", exist_ok=True)
data_url_pattern = re.compile(r"^data:image/\w+;base64,")

# session_id -> frame_counter
session_counters = {}

def new_session_id():
    # e.g. 2025-11-18_15-42-30
    return datetime.now().isoformat().replace("T", "_").replace(":", "-").split(".")[0]


def get_session_id_from_cookie(cookie_header: str | None) -> str | None:
    if not cookie_header:
        return None
    parts = cookie_header.split(";")
    for part in parts:
        name, _, value = part.strip().partition("=")
        if name == "session_id" and value:
            return value
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/start_session":
            self.handle_start_session()
        elif self.path == "/upload_frame":
            self.handle_upload_frame()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_start_session(self):
        # create new session id
        session_id = new_session_id()

        # initialize counter
        session_counters[session_id] = 0

        # ensure folder exists
        session_folder = os.path.join("frames", session_id)
        os.makedirs(session_folder, exist_ok=True)

        # respond with Set-Cookie
        self.send_response(200)
        self.send_header("Set-Cookie", f"session_id={session_id}; Path=/")
        self.end_headers()
        self.wfile.write(b"OK")

        print(f"[SESSION] Started new session: {session_id}")

    def handle_upload_frame(self):
        # get session id from cookie, or auto-create if missing
        cookie_header = self.headers.get("Cookie")
        session_id = get_session_id_from_cookie(cookie_header)

        if not session_id:
            # fallback: create a default session on first frame if none exists
            session_id = new_session_id()
            session_counters[session_id] = 0
            session_folder = os.path.join("frames", session_id)
            os.makedirs(session_folder, exist_ok=True)
            print(f"[SESSION] Implicitly started session: {session_id}")
        else:
            session_folder = os.path.join("frames", session_id)
            os.makedirs(session_folder, exist_ok=True)

        # read body with data URL
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        data_url = body.decode("utf-8")

        # strip 'data:image/jpeg;base64,...'
        b64_data = data_url_pattern.sub("", data_url)
        img_bytes = base64.b64decode(b64_data)

        # frame index for this session
        counter = session_counters.get(session_id, 0)
        filename = os.path.join(session_folder, f"frame_{counter:05d}.jpg")
        session_counters[session_id] = counter + 1

        with open(filename, "wb") as f:
            f.write(img_bytes)

        # respond OK
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

        # optional debug
        # print(f"[FRAME] {session_id} -> {filename}")


# create HTTPS server
httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on https://0.0.0.0:{PORT}")
httpd.serve_forever()
