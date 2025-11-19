# Phone Camera → Laptop Frame Capture

A minimal setup to stream camera frames from a phone to a laptop over HTTPS and save them in timestamped folders. Works on iPhone Safari/Chrome and Android Chrome.

## Contents
- `camera_client.html` — Mobile-friendly page opened on the phone.
- `https_server.py` — HTTPS server that serves the page and receives frames.
- `cert.pem` / `key.pem` — Self-signed TLS certificate and key.
- `frames/` — Automatically contains one folder per recording session.

## Setup
1. (Optional) Create certificates:
   ```bash
   openssl req -new -x509 -keyout key.pem -out cert.pem -nodes -days 365
    ```

2. Start the server:
    ```bash
    python3 https_server.py
    ```

3. On your phone, open:
    ```bash
    https://<LAPTOP-IP>:8443/camera_client.html
    ```
    (Hint: to fetch your IP address) 
    ```bash
    ipconfig getifaddr en0
    ````

4. Setup capture interval (in ms) and press Start streaming. Frames are saved under:
    ```bash
    frames/<timestamp>/frame_00000.jpg
    ```
    Stop after all frames are caught.

