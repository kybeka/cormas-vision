#!/usr/bin/env python3
"""
Evaluate the current model against the corrected ETH sample.

Run AFTER the labels in eval-sample/labels/ have been corrected to ground truth.
Prints overall + per-class precision / recall / mAP (the pawn classes are the
ones to watch).

NOTE: WEIGHTS below is the OLD 12-class model — it cannot score the four new
pawn classes (ids 12-15). Point WEIGHTS at the retrained 16-class model before
trusting these numbers.
"""
from pathlib import Path
from ultralytics import YOLO
import yaml

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "eval-sample"
# newest trained model under 03-training (skips backups) — auto-picks the latest retrain
_cands = [c for c in (HERE.parent / "03-training").rglob("weights/best.pt") if "backup" not in str(c)]
WEIGHTS = max(_cands, key=lambda p: p.stat().st_mtime) if _cands else None
CLASSES = ['blue-pawn', 'blue-token', 'board', 'green-token', 'hand',
           'inner-board', 'orange-token', 'red-pawn', 'red-token',
           'white-pawn', 'yellow-pawn', 'yellow-token',
           'black-pawn', 'green-pawn', 'orange-pawn', 'pink-pawn']


def main():
    data = {"path": str(SAMPLE.resolve()), "train": "images", "val": "images",
            "names": {i: n for i, n in enumerate(CLASSES)}}
    yml = SAMPLE / "_auto.yaml"
    yml.write_text(yaml.safe_dump(data))
    model = YOLO(str(WEIGHTS))
    model.val(data=str(yml), imgsz=640, conf=0.25, iou=0.5, task="obb")


if __name__ == "__main__":
    main()
