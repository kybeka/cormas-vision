#!/usr/bin/env python3
"""
Demo runner for Planet-C detection and cell mapping.

Pipeline:
- Discover latest phone frames folder (phone-server/frames or phone-server/frame)
- Load best available YOLO weights (prefers iter_02)
- For each frame:
  - Run inference
  - Detect board via Canny + largest quadrilateral
  - Compute homography to normalized board space
  - Map detection centers to grid cells (1..N, row-major)
  - Save raw detections to demo_detections/<frame>.txt
  - Print lines: "#green <cells...>" and "#red <cells...>" for present classes

Assumptions:
- Default grid is 4 rows x 5 cols => 20 cells.
- Class names contain "green" or "red" for grouping. Others are ignored in print.
- If board detection fails, mapping is skipped for that frame (still saves detections).

You can override defaults via CLI flags.

Usage examples:
  python demo.py --rows 4 --cols 5 --conf 0.45 --iou 0.5
  python demo.py --frames-dir phone-server/frames/2025-11-18T22-15-00Z
  python demo.py --weights yolo_labelling_v2/pseudo_iterations/iter_02/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parent
PHONE_SERVER_DIR = PROJECT_ROOT / "phone-server"
INFERENCE_TXT_DIR = PROJECT_ROOT / "inference_txt"
BOARD_INSET = 0.03  # shrink the inner-board box onto the painted grid lines

from cv_board_utils import (
    detect_board_edges_and_quad,
    detect_board_quad,
    compute_homography,
    transform_point,
    cell_index,
)
from board_from_model import board_quad_from_result, order_quad, inset_quad, QuadSmoother
from cormas_client import CormasClient


def find_latest_frames_dir(explicit: Optional[Path] = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    base = PHONE_SERVER_DIR / "frames"
    if not base.exists():
        raise FileNotFoundError("Expected phone-server/frames to exist (based on https_server.py)")
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    if not subdirs:
        return base
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest


def resolve_weights_path(explicit: Optional[Path] = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    yl = PROJECT_ROOT / "yolo_labelling_v2"
    candidates = list((yl / "pseudo_iterations").rglob("weights/best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt found under yolo_labelling_v2/pseudo_iterations/**/weights")
    # pick most recent by mtime
    best = max(candidates, key=lambda p: p.stat().st_mtime)
    return best


def list_images(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files

def load_model(weights_path: Path) -> YOLO:
    if YOLO is None:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
    return YOLO(str(weights_path))


def get_detection_centers(result) -> List[Tuple[float, float, int, float]]:
    centers: List[Tuple[float, float, int, float]] = []
    names = getattr(result, "names", None)
    # Prefer boxes centers for stability; OBB may still populate boxes
    boxes = getattr(result, "boxes", None)
    if boxes is not None and hasattr(boxes, "xywh") and hasattr(boxes, "cls") and hasattr(boxes, "conf"):
        xywh = boxes.xywh.cpu().numpy()  # (N,4): x,y,w,h in pixels
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for (x, y, w, h), c, cf in zip(xywh, cls, conf):
            centers.append((float(x), float(y), int(c), float(cf)))
        return centers
    # Fallback: use polygons if present (OBB)
    obb = getattr(result, "obb", None)
    if obb is not None and hasattr(obb, "xyxyxyxy") and hasattr(obb, "cls") and hasattr(obb, "conf"):
        polys = obb.xyxyxyxy.cpu().numpy()  # (N,8): 4 points
        cls = obb.cls.cpu().numpy().astype(int)
        conf = obb.conf.cpu().numpy()
        polys = polys.reshape(-1, 4, 2)
        for poly, c, cf in zip(polys, cls, conf):
            cx = float(np.mean(poly[:, 0]))
            cy = float(np.mean(poly[:, 1]))
            centers.append((cx, cy, int(c), float(cf)))
        return centers
    return centers


def group_cells_by_color(cell_ids_with_cls: List[Tuple[int, int]], class_names: Dict[int, str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {"green": [], "red": []}
    for cell_id, cls_id in cell_ids_with_cls:
        name = class_names.get(cls_id, "").lower()
        if "green" in name:
            groups["green"].append(cell_id)
        elif "red" in name:
            groups["red"].append(cell_id)
    # Deduplicate while preserving order
    for k in list(groups.keys()):
        seen = set()
        dedup = []
        for v in groups[k]:
            if v not in seen:
                dedup.append(v)
                seen.add(v)
        groups[k] = dedup
    return groups


def group_cells_by_class(cell_ids_with_cls: List[Tuple[int, int]], class_names: Dict[int, str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    exclude_substrings = {"board", "hand", "inner-board", "inner_board", "inner board"}
    for cell_id, cls_id in cell_ids_with_cls:
        name = class_names.get(cls_id, "").strip()
        lname = name.lower()
        if any(sub in lname for sub in exclude_substrings):
            continue
        groups.setdefault(name, []).append(cell_id)
    # Deduplicate while preserving order
    for k in list(groups.keys()):
        seen = set()
        dedup = []
        for v in groups[k]:
            if v not in seen:
                dedup.append(v)
                seen.add(v)
        groups[k] = dedup
    return groups


def save_detections_txt(out_dir: Path, frame_name: str, detections: List[Tuple[float, float, int, float]], class_names: Dict[int, str], mapped_cells: Dict[int, int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(frame_name).stem}.txt"
    with path.open("w") as f:
        for x, y, cls_id, conf in detections:
            cname = class_names.get(cls_id, str(cls_id))
            cell = mapped_cells.get((int(x), int(y), cls_id), None)
            # Write: class_name conf x y [cell]
            if cell is not None:
                f.write(f"{cname}\t{conf:.3f}\t{int(x)}\t{int(y)}\t{cell}\n")
            else:
                f.write(f"{cname}\t{conf:.3f}\t{int(x)}\t{int(y)}\n")


def main():
    parser = argparse.ArgumentParser(description="Planet-C demo runner")
    parser.add_argument("--frames-dir", type=str, default=None, help="Path to frames directory (defaults to latest phone-server folder)")
    parser.add_argument("--weights", type=str, default=None, help="YOLO weights path")
    parser.add_argument("--rows", type=int, default=4, help="Board rows (default 4)")
    parser.add_argument("--cols", type=int, default=5, help="Board cols (default 5)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--ws-url", type=str, default=None, help="WebSocket url for CORMAS (e.g., ws://localhost:8081/ws)")
    args = parser.parse_args()

    frames_dir = find_latest_frames_dir(Path(args.frames_dir) if args.frames_dir else None)
    weights_path = resolve_weights_path(Path(args.weights) if args.weights else None)

    print(f"[demo] Frames directory: {frames_dir}")
    print(f"[demo] Weights: {weights_path}")
    print(f"[demo] Grid: {args.rows}x{args.cols}")

    images = list_images(frames_dir)
    if not images:
        print("[demo] No images found in frames directory (watching for new ones...).")

    model = load_model(weights_path)
    # Determine session id from frames_dir and outputs under frames/<session_id>
    session_id = frames_dir.name if frames_dir.name != "frames" else "session"
    out_dir = PROJECT_ROOT / "frames" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start persistent WS client if provided
    client: Optional[CormasClient] = CormasClient(args.ws_url) if args.ws_url else None
    if client:
        client.start()

    print(f"[demo] Watching for new frames in: {frames_dir}")
    processed = set()
    board_smoother = QuadSmoother(0.35)
    try:
        while True:
            images = list_images(frames_dir)
            new_found = False
            for img_path in images:
                if img_path.name in processed:
                    continue
                new_found = True

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[demo] Skipping unreadable image: {img_path}")
                    continue

                # Inference
                results = model.predict(source=img, imgsz=640, conf=args.conf, iou=args.iou, verbose=False)
                if not results:
                    print(f"[demo] No results for {img_path.name}")
                    processed.add(img_path.name)
                    continue
                r = results[0]
                class_names = getattr(r, "names", getattr(model, "names", {}))
                detections = get_detection_centers(r)

                # Render annotated predictions image
                try:
                    annotated = r.plot()
                except Exception:
                    annotated = img.copy()

                # Board detection: prefer the model's inner-board OBB (far steadier than
                # Canny), fall back to Canny, then smooth across frames and inset onto the grid.
                edges_img, canny_quad = detect_board_edges_and_quad(img)
                board_quad = board_quad_from_result(r)
                if board_quad is None and canny_quad is not None:
                    board_quad = order_quad(canny_quad)
                board_quad = board_smoother.update(board_quad)
                if board_quad is not None:
                    board_quad = inset_quad(board_quad, BOARD_INSET)
                    H = compute_homography(board_quad)
                else:
                    H = None
                mapped_cells_with_cls: List[Tuple[int, int]] = []
                mapped_cells_dict: Dict[Tuple[int, int, int], int] = {}
                if H is None:
                    print(f"[demo] Board not detected in {img_path.name}; mapping skipped.")
                else:
                    for x, y, cls_id, conf in detections:
                        xn, yn = transform_point(H, (x, y))
                        cell = cell_index(xn, yn, args.rows, args.cols)
                        mapped_cells_with_cls.append((cell, cls_id))
                        mapped_cells_dict[(int(x), int(y), cls_id)] = cell

                # Save artifacts: annotated image, edges image, and detections txt
                stem = Path(img_path.name).stem
                ann_path = out_dir / f"{stem}_annotated.jpg"
                edge_path = out_dir / f"{stem}_edges.png"
                cv2.imwrite(str(ann_path), annotated)
                cv2.imwrite(str(edge_path), edges_img)
                save_detections_txt(out_dir, img_path.name, detections, class_names, mapped_cells_dict)

                # Group and print per class (no hardcoded colors)
                class_groups = group_cells_by_class(mapped_cells_with_cls, class_names)
                print(f"[frame] {img_path.name}")
                for cname, cells in class_groups.items():
                    if cells:
                        print(f"#{cname} " + " ".join(str(c) for c in cells))

                # Append the class outputs to the same per-frame txt
                txt_path = out_dir / f"{stem}.txt"
                try:
                    with open(txt_path, "a") as f:
                        for cname, cells in class_groups.items():
                            if cells:
                                f.write(f"#{cname} " + " ".join(str(c) for c in cells) + "\n")
                except Exception as e:
                    print(f"[demo] Failed to append outputs to {txt_path.name}: {e}")

                # Optional: send to CORMAS via persistent WebSocket
                if client:
                    try:
                        client.send_class_map(class_groups)
                    except Exception as e:
                        print(f"[demo] WS send failed: {e}")

                processed.add(img_path.name)

            # If no new files, short sleep to avoid busy waiting
            if not new_found:
                time.sleep(0.5)
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()