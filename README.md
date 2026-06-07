# cormas-vision

Real-time computer vision for physical board games, feeding into **CORMAS** — the agent-based modelling platform built in Pharo. Part of Google Summer of Code 2025 with the Pharo Consortium and CIRAD.

The idea: point a camera at a physical board, detect the pieces, map them to grid cells, and push that state into a CORMAS simulation — so a tangible board drives the model. Chess was the prototype; the focus narrowed to **Planet-C**, a CIRAD/LEAF serious game played on a 5×4 grid with coloured planet tokens.

## How the repo is laid out

It's organised as the project actually unfolded, phase by phase:

| Phase | Folder | What it is |
|-------|--------|------------|
| 0 | `docs/` | Proposal, timeline, weekly reports, pipeline notes |
| 1 | `01-proof-of-concept/` | Earliest chess prototypes (RTMP + USB streaming + YOLO) |
| 2 | `02-exploration/` | Auto-labelling experiments with GroundingDINO and SAM — tried, then dropped |
| 3 | `03-training/` | The real pipeline: YOLO11-OBB training + iterative pseudo-labelling + the inspector |
| 4 | `04-eth-trial/` | Live ETH trial (Nov 2025): logs + findings |
| 5 | `05-runtime/` | The live demo: phone capture → detection → board homography → cell mapping |
| 6 | `06-cormas-bridge/` | Connecting it all to CORMAS (Pharo) — still to build |

See `docs/` for the proposal, timeline, and weekly reports.

## The pipeline (stages 1–6 work; stage 7 is the gap)

1. **Capture** — phone camera streams frames to the laptop over HTTPS (`05-runtime/phone-server`).
2. **Board detection + homography** — find the board, rectify to a normalised top-down view.
3. **Detection** — YOLO11-OBB locates pieces (oriented boxes handle rotation).
4. **Cell mapping** — each detection's centre maps to a grid cell (1..20).
5. **State** — per-frame class → cells, written to disk and (optionally) streamed.
6. **CORMAS** — *to build:* a Pharo side that consumes the state (WebSocket **and** JSON variants planned).

## Running the demo

```bash
# 1) start the phone capture server (serves the camera page + receives frames)
cd 05-runtime/phone-server && python3 https_server.py

# 2) on your phone, open https://<LAPTOP-IP>:8443/camera_client.html and start streaming

# 3) run inference + cell mapping over the captured frames
cd 05-runtime && python3 demo.py --rows 4 --cols 5
```

## Status

Detection is strong for the well-labelled classes (green/yellow tokens, board) and weak or missing for the rest — a data-coverage gap, not a model failure. The CORMAS bridge is the main remaining piece.

## Heavy data

Datasets, model weights, videos and training runs are kept out of git (Roboflow holds the annotated images).
