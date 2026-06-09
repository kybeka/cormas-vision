# Pharo / Cormas side — where it lives

The Cormas (Pharo) half of this project looked "unbuilt" in this repo (only an empty
`pharo/CormasVision/FirstClass` stub), but it actually exists in local Pharo **images**
outside git: `~/Documents/Pharo/images/`. The `.changes` log of the socket images
contains `ZnWebSocket` + the `bonjour` handshake — i.e. the **Pharo WebSocket server**
side of the exact protocol `../05-runtime/cormas_client.py` and `ws-prototype/client.py`
speak to (`ws://localhost:8081/ws`).

## The images (as of 2026-06)
| Image | Date | What it is |
|-------|------|------------|
| `PlanetCModel` | Nov 3 | The Planet-C **Cormas agent-based model** |
| `PlanetCwSockets` | Nov 17 | Planet-C model **+ the ZnWebSocket bridge** (the integration) |
| `SocketCommunication` | Nov 19 | Socket bridge work (ZnWebSocket + bonjour) |
| `TestSocketCommunication` | Nov 19 | Tests for the socket bridge |
| `Issue#829`, `Issue#833` | Aug 27 | Cormas framework bug-fix contributions |
| `cormas-vision-Pharo12` | Jun 26 | Early GSoC repo image |
| `CORMAS Pharo13 (latest dev)` | Sep 30 | Cormas dev image |
| `Pharo13 (stable)` | Nov 15 | Base image |
| `MOOC` | Jan | Pharo MOOC (learning) |

Each image is ~300–400 MB (`.image` + `.changes` + `.sources`) — too big to commit, and
they're working images, not source.

## TODO: recover the code into the repo
The real deliverable is the **source**, not the images. In Pharo (open `PlanetCwSockets`),
use **Iceberg / Tonel export** (or file-out the package) to write the Planet-C model + the
`ZnWebSocket` server package into `pharo/` here, replacing the `FirstClass` stub. That puts
the Cormas bridge under version control and connects it to the Python side
(`ws-prototype/`, `../05-runtime/cormas_client.py`, `PROTOCOL.md`). Requires running Pharo,
so it's a you-step — I can't drive the image from here.

## `ws-prototype/`
The Python WebSocket bridge prototype recovered from `~/Documents/PlanetC`: `client.py`
sends board state (`{"blue-pawn":[8,10], ...}`) to the Pharo server after a `bonjour`
handshake; `server.py` is a standalone test server; `mock.py` simulates occupied cells.
