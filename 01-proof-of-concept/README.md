# 01 — Proof of concept (chess)

The earliest prototypes, using **chess** as a stand-in before narrowing to Planet-C. Goal: prove the camera → detection → live-stream loop end to end.

- `chess-rmtp-streaming/` — Dockerised pipeline that ingests an RTMP stream (phone → laptop) and runs YOLO chess-piece detection. Demo frame captures live under `output/` (git-ignored).
- `chess-usb-inference/` — the same idea over a wired USB camera, which proved more reliable than RTMP for latency (see `../docs/weekly-reports/week-04_2025-07-14_streaming-options.md`).

These established the capture + inference loop that the Planet-C runtime in `../05-runtime` later replaced with the phone HTTPS server.
