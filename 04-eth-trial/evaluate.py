#!/usr/bin/env python3
"""
Evaluate the newest trained model on the frozen 26-frame ETH test set and log the
result to MLflow, so retrains are directly comparable.

The number that matters across experiments is the TEST-set per-class mAP (training
only logs valid-set metrics) — so this is what we track. Each run lands in the
`planet-c-detection` experiment under `03-training/mlruns`.

  python evaluate.py
  mlflow ui --backend-store-uri ../03-training/mlruns --port 5050
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "eval-sample"
MLRUNS = HERE.parent / "03-training" / "mlruns"
os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{MLRUNS}")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "planet-c-detection")

import yaml
import mlflow
from ultralytics import YOLO, settings

settings.update({"mlflow": False})  # we log explicitly; avoid ultralytics double-logging

# newest trained model under 03-training (skips backups) — auto-picks the latest retrain
_cands = [c for c in (HERE.parent / "03-training").rglob("weights/best.pt") if "backup" not in str(c)]
WEIGHTS = max(_cands, key=lambda p: p.stat().st_mtime) if _cands else None
CLASSES = ['blue-pawn', 'blue-token', 'board', 'green-token', 'hand',
           'inner-board', 'orange-token', 'red-pawn', 'red-token',
           'white-pawn', 'yellow-pawn', 'yellow-token',
           'black-pawn', 'green-pawn', 'orange-pawn', 'pink-pawn']


def main():
    assert WEIGHTS, "no trained model found under 03-training"
    yml = SAMPLE / "_auto.yaml"
    yml.write_text(yaml.safe_dump({"path": str(SAMPLE.resolve()), "train": "images", "val": "images",
                                   "names": {i: n for i, n in enumerate(CLASSES)}}))
    m = YOLO(str(WEIGHTS)).val(data=str(yml), imgsz=640, conf=0.25, iou=0.5, task="obb")

    run_name = WEIGHTS.parents[2].name if len(WEIGHTS.parents) > 2 else WEIGHTS.stem
    with mlflow.start_run(run_name=f"{run_name} · test"):
        mlflow.log_params({"weights": str(WEIGHTS), "test_set": "eth-26", "conf": 0.25})
        for k, v in m.results_dict.items():
            try:
                mlflow.log_metric(k.replace("metrics/", "").replace("(B)", ""), float(v))
            except Exception:
                pass
        try:  # per-class mAP50 — the pawn classes are what we watch
            for i, ci in enumerate(m.box.ap_class_index):
                mlflow.log_metric(f"mAP50/{m.names[int(ci)]}", float(m.box.ap50[i]))
        except Exception as e:
            print(f"[eval] per-class log skipped: {e}")
    print(f"\n[eval] logged '{run_name} · test' to MLflow experiment 'planet-c-detection'")


if __name__ == "__main__":
    main()
