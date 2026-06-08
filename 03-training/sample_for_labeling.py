#!/usr/bin/env python3
"""
Build a high-value labelling batch for the inspector.

Two ideas to cut labelling time:
  1. Sample frames SPREAD across a video/session (distinct board states), not
     consecutive near-duplicates.
  2. PRE-SEED labels: a robust static board (the median inner-board across the
     sampled frames, so you don't fix the board every frame) + the model's piece
     predictions. You then just correct/add pawns.

Writes into pseudo_iterations/iter_<N>/{pseudo_images,pseudo_labels} so you can
open it directly:  (from 03-training/)  python pseudo_labeling/simple_inspector.py -i <N>

Usage:
  python sample_for_labeling.py --video game-data/videos/source-02_oleks.TS.mp4 --n 40 --iter 91
  python sample_for_labeling.py --frames-dir ../05-runtime/phone-server/frames/<session> --n 30 --iter 92
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05-runtime"))
from board_from_model import board_quad_from_result, order_quad  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_W = HERE / "pseudo_iterations/iter_02/weights/best.pt"
INNER_BOARD_ID = 5  # class id in the schema


def select_from_video(video: Path, n: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, img = cap.read()
        if ok:
            frames.append((f"frame_{int(i):06d}", img))
    cap.release()
    return frames


def select_from_dir(d: Path, n: int):
    files = sorted(d.glob("*.jpg"))
    idxs = np.linspace(0, len(files) - 1, min(n, len(files)), dtype=int)
    return [(files[i].stem, cv2.imread(str(files[i]))) for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str)
    ap.add_argument("--frames-dir", type=str)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--weights", type=str, default=str(DEFAULT_W))
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    if args.video:
        frames = select_from_video(Path(args.video), args.n)
    elif args.frames_dir:
        frames = select_from_dir(Path(args.frames_dir), args.n)
    else:
        ap.error("pass --video or --frames-dir")

    out = HERE / "pseudo_iterations" / f"iter_{args.iter:02d}"
    (out / "pseudo_images").mkdir(parents=True, exist_ok=True)
    (out / "pseudo_labels").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results, board_quads = [], []
    for name, img in frames:
        r = model.predict(source=img, imgsz=640, conf=args.conf, verbose=False)[0]
        results.append((name, img, r))
        bq = board_quad_from_result(r)
        if bq is not None:
            board_quads.append(bq)

    # robust static board = median corner across the sampled frames (board doesn't move)
    board = order_quad(np.median(np.stack(board_quads), axis=0)) if board_quads else None

    for name, img, r in results:
        h, w = img.shape[:2]
        cv2.imwrite(str(out / "pseudo_images" / f"{name}.jpg"), img)
        lines = []
        if board is not None:  # pre-seed the inner-board so you don't redraw it each frame
            coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in board)
            lines.append(f"{INNER_BOARD_ID} {coords}")
        if r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            cls = r.obb.cls.cpu().numpy().astype(int)
            for poly, c in zip(polys, cls):
                if c == INNER_BOARD_ID:
                    continue  # already seeded the robust board
                coords = " ".join(f"{x/w:.6f} {y/h:.6f}" for x, y in poly)
                lines.append(f"{int(c)} {coords}")
        (out / "pseudo_labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")

    print(f"wrote {len(results)} frames to {out}")
    print(f"board pre-seeded: {board is not None}  (from {len(board_quads)} detections)")
    print(f"inspect with:  cd {HERE.name} && python pseudo_labeling/simple_inspector.py -i {args.iter}")


if __name__ == "__main__":
    main()
