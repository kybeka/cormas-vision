#!/usr/bin/env python3
"""
Evaluate the current model against the corrected ETH sample.

Run AFTER the labels in eval-sample/labels/ have been corrected to ground truth.
Prints overall + per-class precision / recall / mAP (the pawn classes are the
ones to watch).
"""
from pathlib import Path
from ultralytics import YOLO
import yaml

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "eval-sample"
WEIGHTS = HERE.parent / "03-training/pseudo_iterations/iter_02/weights/best.pt"
CLASSES = ['blue-pawn', 'blue-token', 'board', 'green-token', 'hand',
           'inner-board', 'orange-token', 'red-pawn', 'red-token',
           'white-pawn', 'yellow-pawn', 'yellow-token']


def main():
    data = {"path": str(SAMPLE), "val": "images",
            "names": {i: n for i, n in enumerate(CLASSES)}}
    yml = SAMPLE / "_auto.yaml"
    yml.write_text(yaml.safe_dump(data))
    model = YOLO(str(WEIGHTS))
    model.val(data=str(yml), imgsz=640, conf=0.25, iou=0.5, task="obb")


if __name__ == "__main__":
    main()
