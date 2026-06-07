# 04 — ETH trial (Nov 2025)

Live test of the Planet-C pipeline at ETH. A half-pseudo-labelled YOLO11-OBB model was run on phone-captured frames of a real game, with board detection + cell mapping.

## What's here
- `logs/` — the per-frame demo outputs (`<session>/frame_NNNNN.txt`): one line per detection (`class  conf  x  y  cell`) plus the grouped `#class cell cell ...` summary. Three sessions, 777 frames total.
- Result videos (`annotated.mp4`, `edges.mp4`, `side_by_side.mp4`) and raw trial photos are kept out of git for now — see `../CONSOLIDATION_PLAN.md` (data relocation pending).

## What the logs show (see `../PROJECT_STATE.md` §3 for the full table)
- **Strong:** green-token (~0.90 conf), yellow-token (~0.94), board / inner-board (~0.93–0.95).
- **Weak:** orange-token (~0.68), red-pawn (~0.60), blue-pawn (~0.48), hand (~0.65).
- **Never detected:** blue-token, red-token, yellow-pawn, white-pawn — the model had no usable labels for these.
- **Stability bug:** the board quad is re-detected every frame, so the homography wobbles and stationary tokens hop cells (203 distinct green-token cell-sets across 648 frames).

## Next (Phase 1)
Annotate a small stratified sample (~30–50 frames, biased to the rare classes) to get a real mAP, and use those frames as real-distribution training data.
