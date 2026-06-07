# 06 — CORMAS bridge (to build)

The piece that connects the vision output to a CORMAS (Pharo) simulation. This is the headline GSoC deliverable and is currently **unbuilt** — `pharo/` holds only an empty `CormasVision` package stub.

## What exists
- `pharo/CormasVision/` — Tonel source of the (empty) `FirstClass` placeholder. The real package goes here.

## Plan — two parallel solutions
1. **WebSocket** (refine the existing client). The runtime already has a Python `CormasClient` (`../05-runtime/cormas_client.py`) that connects to `ws://localhost:8081/ws` and sends `{ "green-token": [6, 11], ... }`. The missing half is a Pharo WebSocket **server** (e.g. `ZnWebSocket`) that consumes those messages and updates the simulation grid.
2. **JSON file-passing** (the proposal's original idea). Python writes board state to a known directory as JSON per frame; Pharo watches and ingests it. Simpler, no live socket, easier to debug.

## Message shape (current)
`demo.py` emits, per frame, a map of class name → list of occupied cell indices (1..20, row-major on the Planet-C 5×4 grid). The bridge maps those cells onto CORMAS spatial entities.
