import http.server
import socketserver
import ssl
import os
import re
import base64

PORT = 8443
DIRECTORY = "."

os.makedirs("frames", exist_ok=True)
data_url_pattern = re.compile(r"^data:image/\w+;base64,")
frame_counter = 0


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        global frame_counter

        if self.path == "/upload_frame":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            data_url = body.decode("utf-8")

            # strip 'data:image/jpeg;base64,...'
            b64_data = data_url_pattern.sub("", data_url)
            img_bytes = base64.b64decode(b64_data)

            filename = f"frames/frame_{frame_counter:05d}.jpg"
            with open(filename, "wb") as f:
                f.write(img_bytes)
            frame_counter += 1

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()


# create HTTPS server
httpd = socketserver.TCPServer(("0.0.0.0", PORT), Handler)

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

print(f"Serving HTTPS on https://0.0.0.0:{PORT}")
httpd.serve_forever()
