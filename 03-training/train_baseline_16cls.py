#!/usr/bin/env python3
"""
First real 16-class baseline.

Trains YOLO11-OBB on the existing labelled frames (old-schema labels are valid
under the additive 16-class data.yaml). Heavier rotation + HSV augmentation than
before, to test whether augmentation helps the tilted-board / lighting issues.

Expected: good on tokens, ~0 on pawns (no pawn labels in this training data) —
this is the BASELINE the Oleks labelling round is measured against.
Evaluate afterwards with 04-eth-trial/evaluate.py pointed at the new weights.
"""
import os
from pathlib import Path
import torch
from ultralytics import YOLO

os.chdir(Path(__file__).resolve().parent)
# Consolidate every training run into ONE MLflow experiment so retrains are
# directly comparable (ultralytics auto-logs when mlflow is installed).
# View with:  mlflow ui --backend-store-uri 03-training/mlruns
os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{Path.cwd() / 'mlruns'}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "planet-c-detection")
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

model = YOLO("yolo11m-obb.pt")
results = model.train(
    data="frames.v3i.yolov11/data.yaml",
    project="runs/baseline_16cls", name="train", exist_ok=True,
    epochs=80, imgsz=640, batch=8, patience=20, device=device,
    # augmentation: HSV (lighting robustness) + scale/translate/flip.
    # NOTE: degrees>0 (rotation aug) triggers a known ultralytics OBB assigner crash
    # on dense rotated boxes (shape mismatch in tal.py) — keep it at 0. Tilted-board
    # robustness comes from real data (ETH/diverse frames), not rotation aug.
    degrees=0.0, scale=0.5, translate=0.1, fliplr=0.5,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    dropout=0.1, plots=True, verbose=True,
)
print(f"BASELINE DONE. best weights: {results.save_dir}/weights/best.pt")
