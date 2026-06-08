# CORMAS bridge protocol

Two transports carry the **same** board state from the vision runtime to CORMAS:

1. **WebSocket** (live) — `05-runtime/cormas_client.py` connects to `ws://localhost:8081/ws`
   and sends the `classes` map per frame. Needs a Pharo `ZnWebSocket` server (to build).
2. **JSON file** (simple) — `05-runtime/json_bridge.py` atomically overwrites
   `<out-dir>/<session>.json` each frame. The Pharo side polls that one file.
   Enable with `python demo.py --json-out <dir>`.

## State shape (`planet-c/v1`)

```json
{
  "schema": "planet-c/v1",
  "session": "2025-11-21_08-53-26",
  "frame": "frame_00090",
  "timestamp": "2026-06-08T01:30:00",
  "grid": { "rows": 5, "cols": 4, "origin": "cell 1 = A1 = top-left, row-major" },
  "classes": { "green-token": {"1":1,"5":3,"12":1}, "yellow-token": {"3":1,"9":2}, "blue-pawn": {"7":1} },
  "semantics": {
    "biomass":     {"1":1,"5":3,"12":1},
    "birds":       {"3":1,"9":2},
    "non_harvest": {},
    "park_limit":  {},
    "harvesters":  [ { "player": "blue", "cells": {"7":1} } ]
  }
}
```

- **Cells** are `1..20`, row-major on the 5×4 grid: `A1=1, A2=2, A3=3, A4=4, B1=5, … E4=20`.
- **`classes`** = raw detector output as **`{cell: count}`** — multiplicity is kept (a cell
  can hold several pieces, e.g. 3 biomass in cell 5), since the *quantity* is what Planet-C
  resource dynamics care about, not mere presence.
- **`semantics`** = the same data mapped to Planet-C meaning, which is what a CORMAS
  model actually wants: biomass / birds / non-harvest / park-limit tokens, and
  harvesters (the standing pawns) keyed by player colour.

## Why two transports
The WebSocket gives low-latency live updates; the JSON file is trivial to consume and
debug (no socket, just read a file) and survives restarts. The Pharo side can implement
either first — the payload is identical.
