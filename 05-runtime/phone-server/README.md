# Phone Camera → Laptop Frame Capture

A minimal setup to stream camera frames from a phone to a laptop over HTTPS and save them in timestamped folders. Works on iPhone Safari/Chrome and Android Chrome.

## Two servers
- **`https_server.py`** — *capture only*: saves each frame to `frames/<session>/frame_NNNNN.jpg`. Pair it with `demo.py` (which polls that folder). Simple, good for collecting datasets.
- **`inference_server.py`** — *integrated, low-latency*: decodes each frame in memory and runs the `Pipeline` directly on a latest-frame-wins worker (no disk, no polling, stale frames dropped). The live board state is returned on every upload and served at **`GET /state`**. Optional `--json-out <dir>` (CORMAS file bridge) and `--ws-url` (live WebSocket).
  ```bash
  python3 inference_server.py --json-out ../cormas_state
  ```

## Contents
- `camera_client.html` — Mobile-friendly page opened on the phone (works with either server).
- `cert.pem` / `key.pem` — Self-signed TLS certificate and key.
- `frames/` — Per-session frame folders (written by `https_server.py`).

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

