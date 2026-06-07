#!/usr/bin/env python3
"""
Build the ETH evaluation sample.

Picks a stratified set of dense gameplay frames from the main ETH session,
copies the images, and writes YOLO-OBB *pre-labels* from the current model so
they can be corrected (rather than labelled from scratch). After correction
these become ground truth for `model.val()`.
"""
from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SESSION = "2025-11-21_08-53-26"
IMAGES_SRC = REPO / "05-runtime/phone-server/frames" / SESSION
LOGS = HERE / "logs" / SESSION
WEIGHTS = REPO / "03-training/pseudo_iterations/iter_02/weights/best.pt"

OUT = HERE / "eval-sample"
OUT_IMG = OUT / "images"
OUT_LBL = OUT / "labels"

CLASSES = ['blue-pawn', 'blue-token', 'board', 'green-token', 'hand',
           'inner-board', 'orange-token', 'red-pawn', 'red-token',
           'white-pawn', 'yellow-pawn', 'yellow-token',
           'black-pawn', 'green-pawn', 'orange-pawn', 'pink-pawn']


def select_frames():
    all_imgs = sorted(IMAGES_SRC.glob("frame_*.jpg"))
    # frames flagged with the rare blue-pawn (model's own detections)
    blue = {p.stem for p in LOGS.glob("*.txt")
            if any(l.startswith("blue-pawn\t") for l in p.read_text().splitlines())}
    blue_frames = [p for p in all_imgs if p.stem in blue]
    # even spread across active gameplay (skip the first ~30 setup frames)
    active = [p for p in all_imgs if int(p.stem.split("_")[1]) >= 30]
    step = max(1, len(active) // 18)
    spread = active[::step]
    # union, keep order, cap ~26
    chosen, seen = [], set()
    for p in spread + blue_frames:
        if p.stem not in seen:
            chosen.append(p); seen.add(p.stem)
    return sorted(chosen, key=lambda p: p.name)[:26]


def main():
    assert WEIGHTS.exists(), f"weights not found: {WEIGHTS}"
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_LBL.mkdir(parents=True, exist_ok=True)
    frames = select_frames()
    print(f"selected {len(frames)} frames")
    model = YOLO(str(WEIGHTS))
    for p in frames:
        shutil.copy2(p, OUT_IMG / p.name)
        img = cv2.imread(str(p)); h, w = img.shape[:2]
        r = model.predict(source=str(p), imgsz=640, conf=0.25, iou=0.5, verbose=False)[0]
        lines = []
        if r.obb is not None and len(r.obb) > 0:
            polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            cls = r.obb.cls.cpu().numpy().astype(int)
            for poly, c in zip(polys, cls):
                coords = []
                for x, y in poly:
                    coords += [f"{x / w:.6f}", f"{y / h:.6f}"]
                lines.append(f"{c} " + " ".join(coords))
        (OUT_LBL / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"images -> {OUT_IMG}")
    print(f"pre-labels -> {OUT_LBL}")


if __name__ == "__main__":
    main()
