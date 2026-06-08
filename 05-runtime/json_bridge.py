"""
Write Planet-C board state to JSON for the CORMAS bridge (file-passing variant).

The runtime calls write_state() once per frame; it atomically overwrites a single
<session>.json file holding the latest board state. The Pharo/CORMAS side polls
that file. This is the simpler of the two bridge solutions (the other is the live
WebSocket in cormas_client.py); both carry the same board state.
"""
from __future__ import annotations
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Flat-token class -> Planet-C meaning (see the rulebook component list).
TOKEN_SEMANTICS = {
    "green-token": "biomass",      # renewable resource (60 pieces)
    "yellow-token": "birds",       # newborn birds (20)
    "orange-token": "non_harvest", # non-harvest marker (15)
    "red-token": "park_limit",     # park limits (3)
}


def _semantics(class_groups: Dict[str, List[int]]) -> dict:
    out = {"biomass": [], "birds": [], "non_harvest": [], "park_limit": [], "harvesters": []}
    harvesters: Dict[str, List[int]] = {}
    for cls, cells in class_groups.items():
        if cls in TOKEN_SEMANTICS:
            out[TOKEN_SEMANTICS[cls]] = sorted(cells)
        elif cls.endswith("-pawn"):          # harvesters, keyed by player colour
            harvesters[cls[:-5]] = sorted(cells)
    out["harvesters"] = [{"player": c, "cells": harvesters[c]} for c in sorted(harvesters)]
    return out


def build_state(session: str, frame: str, class_groups: Dict[str, List[int]],
                rows: int = 5, cols: int = 4) -> dict:
    return {
        "schema": "planet-c/v1",
        "session": session,
        "frame": frame,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "grid": {"rows": rows, "cols": cols, "origin": "cell 1 = A1 = top-left, row-major"},
        "classes": {k: sorted(v) for k, v in class_groups.items() if v},
        "semantics": _semantics(class_groups),
    }


def write_state(out_dir, session: str, frame: str, class_groups: Dict[str, List[int]],
                rows: int = 5, cols: int = 4) -> Path:
    """Atomically write the latest board state to <out_dir>/<session>.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_state(session, frame, class_groups, rows, cols)
    dest = out_dir / f"{session}.json"
    fd, tmp = tempfile.mkstemp(dir=str(out_dir), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, dest)   # atomic: the poller never sees a half-written file
    return dest
