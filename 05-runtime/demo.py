#!/usr/bin/env python3
"""
Demo runner for Planet-C detection and cell mapping.

Per frame: YOLO-OBB inference -> board homography from the model's inner-board
OBB (smoothed across frames, Canny fallback) -> map piece centres to grid cells
(5x4, cell 1 = A1 = inner-board top-left) -> emit artifacts (annotated / edges /
detections txt), a per-cell-count JSON for the CORMAS file bridge, and/or a live
WebSocket message.

Usage:
  python demo.py                                  # watch the latest phone-server session
  python demo.py --frames-dir <dir> --once        # process a folder once and exit
  python demo.py --json-out cormas_state          # also write the CORMAS JSON bridge
  python demo.py --ws-url ws://localhost:8081/ws   # also stream over WebSocket (CMVisionServerCommand default)
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from cv_board_utils import detect_board_edges_and_quad, compute_homography, transform_point, cell_index
from board_from_model import board_quad_from_result, order_quad, inset_quad, QuadSmoother
from json_bridge import write_state
from cormas_client import CormasClient

PROJECT_ROOT = Path(__file__).resolve().parent
PHONE_SERVER_DIR = PROJECT_ROOT / "phone-server"
TRAINING_DIR = PROJECT_ROOT.parent / "03-training"
BOARD_INSET = 0.03  # shrink the inner-board box onto the painted grid lines
EXCLUDE = {"board", "hand", "inner-board", "inner_board", "inner board"}

Detection = Tuple[float, float, int, float]  # x, y, class_id, conf


# --------------------------------------------------------------------------- io
def find_latest_frames_dir(explicit: Optional[Path] = None) -> Path:
    if explicit and explicit.exists():
        return explicit
    base = PHONE_SERVER_DIR / "frames"
    if not base.exists():
        raise FileNotFoundError("Expected phone-server/frames to exist (based on https_server.py)")
    subdirs = [d for d in base.iterdir() if d.is_dir()]
    return max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else base


def resolve_weights_path(explicit: Optional[Path] = None) -> Path:
    """Newest best.pt under 03-training (skips backups) — picks up the latest retrain."""
    if explicit and explicit.exists():
        return explicit
    candidates = [c for c in TRAINING_DIR.rglob("weights/best.pt") if "backup" not in str(c)]
    if not candidates:
        raise FileNotFoundError(f"No weights/best.pt found under {TRAINING_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def list_images(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted((p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts),
                  key=lambda p: p.name)


# ----------------------------------------------------- pure detection / mapping
def get_detection_centers(result) -> List[Detection]:
    """Centre (x, y, class_id, conf) of each detection, preferring boxes then OBB polys."""
    boxes = getattr(result, "boxes", None)
    if boxes is not None and all(hasattr(boxes, a) for a in ("xywh", "cls", "conf")):
        xywh, cls, conf = boxes.xywh.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()
        return [(float(x), float(y), int(c), float(cf)) for (x, y, w, h), c, cf in zip(xywh, cls, conf)]
    obb = getattr(result, "obb", None)
    if obb is not None and all(hasattr(obb, a) for a in ("xyxyxyxy", "cls", "conf")):
        polys = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        cls, conf = obb.cls.cpu().numpy().astype(int), obb.conf.cpu().numpy()
        return [(float(p[:, 0].mean()), float(p[:, 1].mean()), int(c), float(cf)) for p, c, cf in zip(polys, cls, conf)]
    return []


def map_detections_to_cells(detections: List[Detection], H, rows: int, cols: int
                            ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int, int], int]]:
    """-> ([(cell, class_id)], {(x,y,class_id): cell}). Empty if H is None."""
    mapped: List[Tuple[int, int]] = []
    mapped_dict: Dict[Tuple[int, int, int], int] = {}
    if H is None:
        return mapped, mapped_dict
    for x, y, cls_id, conf in detections:
        xn, yn = transform_point(H, (x, y))
        cell = cell_index(xn, yn, rows, cols)
        mapped.append((cell, cls_id))
        mapped_dict[(int(x), int(y), cls_id)] = cell
    return mapped, mapped_dict


def _dedup(seq: List[int]) -> List[int]:
    seen, out = set(), []
    for v in seq:
        if v not in seen:
            out.append(v); seen.add(v)
    return out


def group_cells_by_class(mapped: List[Tuple[int, int]], class_names: Dict[int, str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for cell, cls_id in mapped:
        name = class_names.get(cls_id, "").strip()
        if any(sub in name.lower() for sub in EXCLUDE):
            continue
        groups.setdefault(name, []).append(cell)
    return {k: _dedup(v) for k, v in groups.items()}


def cell_counts(mapped: List[Tuple[int, int]], class_names: Dict[int, str]) -> Dict[str, Dict[int, int]]:
    """{class_name: {cell: count}} — multiplicity kept (board/hand/inner-board excluded)."""
    counts: Dict[str, Dict[int, int]] = {}
    for cell, cls_id in mapped:
        name = class_names.get(cls_id, str(cls_id)).strip()
        if name.lower() in EXCLUDE:
            continue
        counts.setdefault(name, {})[cell] = counts.setdefault(name, {}).get(cell, 0) + 1
    return counts


def save_detections_txt(out_dir: Path, frame_name: str, detections: List[Detection],
                        class_names: Dict[int, str], mapped_dict: Dict[Tuple[int, int, int], int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{Path(frame_name).stem}.txt").open("w") as f:
        for x, y, cls_id, conf in detections:
            cname = class_names.get(cls_id, str(cls_id))
            cell = mapped_dict.get((int(x), int(y), cls_id))
            tail = f"\t{cell}" if cell is not None else ""
            f.write(f"{cname}\t{conf:.3f}\t{int(x)}\t{int(y)}{tail}\n")


# ------------------------------------------------------------------- pipeline
@dataclass
class FrameOutcome:
    class_names: Dict[int, str]
    detections: List[Detection]
    annotated: np.ndarray
    edges_img: np.ndarray
    H: Optional[np.ndarray]
    mapped: List[Tuple[int, int]] = field(default_factory=list)
    mapped_dict: Dict[Tuple[int, int, int], int] = field(default_factory=dict)


class Pipeline:
    """Capture -> infer -> board homography -> map to cells -> emit."""

    def __init__(self, weights: Path, rows: int = 5, cols: int = 4, conf: float = 0.45,
                 iou: float = 0.5, inset: float = BOARD_INSET,
                 json_out: Optional[str] = None, ws_url: Optional[str] = None):
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics")
        self.model = YOLO(str(weights))
        self.rows, self.cols, self.conf, self.iou, self.inset = rows, cols, conf, iou, inset
        self.json_out = json_out
        self.smoother = QuadSmoother(0.35)
        self.client: Optional[CormasClient] = CormasClient(ws_url) if ws_url else None
        if self.client:
            self.client.start()

    def board_homography(self, img, result) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Prefer the model's inner-board OBB (steady), Canny fallback, smooth + inset."""
        edges_img, canny_quad = detect_board_edges_and_quad(img)
        quad = board_quad_from_result(result)
        if quad is None and canny_quad is not None:
            quad = order_quad(canny_quad)
        quad = self.smoother.update(quad)
        if quad is None:
            return None, edges_img
        return compute_homography(inset_quad(quad, self.inset)), edges_img

    def process(self, img) -> Optional[FrameOutcome]:
        results = self.model.predict(source=img, imgsz=640, conf=self.conf, iou=self.iou, verbose=False)
        if not results:
            return None
        r = results[0]
        class_names = getattr(r, "names", getattr(self.model, "names", {}))
        detections = get_detection_centers(r)
        try:
            annotated = r.plot()
        except Exception:
            annotated = img.copy()
        H, edges_img = self.board_homography(img, r)
        mapped, mapped_dict = map_detections_to_cells(detections, H, self.rows, self.cols)
        return FrameOutcome(class_names, detections, annotated, edges_img, H, mapped, mapped_dict)

    def emit(self, out_dir: Path, session_id: str, frame_name: str, o: FrameOutcome) -> None:
        stem = Path(frame_name).stem
        cv2.imwrite(str(out_dir / f"{stem}_annotated.jpg"), o.annotated)
        cv2.imwrite(str(out_dir / f"{stem}_edges.png"), o.edges_img)
        save_detections_txt(out_dir, frame_name, o.detections, o.class_names, o.mapped_dict)

        groups = group_cells_by_class(o.mapped, o.class_names)
        print(f"[frame] {frame_name}")
        for cname, cells in groups.items():
            if cells:
                print(f"#{cname} " + " ".join(map(str, cells)))
        try:
            with (out_dir / f"{stem}.txt").open("a") as f:
                for cname, cells in groups.items():
                    if cells:
                        f.write(f"#{cname} " + " ".join(map(str, cells)) + "\n")
        except Exception as e:
            print(f"[demo] Failed to append outputs: {e}")

        if self.json_out:
            try:
                write_state(self.json_out, session_id, stem, cell_counts(o.mapped, o.class_names), self.rows, self.cols)
            except Exception as e:
                print(f"[demo] JSON write failed: {e}")
        if self.client:
            try:
                self.client.send_frame(groups)
            except Exception as e:
                print(f"[demo] WS send failed: {e}")

    def run_once(self, frames_dir: Path, out_dir: Path, session_id: str) -> None:
        for img_path in list_images(frames_dir):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[demo] Skipping unreadable image: {img_path}")
                continue
            o = self.process(img)
            if o is None:
                print(f"[demo] No results for {img_path.name}")
                continue
            if o.H is None:
                print(f"[demo] Board not detected in {img_path.name}; mapping skipped.")
            self.emit(out_dir, session_id, img_path.name, o)

    def watch(self, frames_dir: Path, out_dir: Path, session_id: str) -> None:
        print(f"[demo] Watching for new frames in: {frames_dir}")
        processed: set = set()
        while True:
            new_found = False
            for img_path in list_images(frames_dir):
                if img_path.name in processed:
                    continue
                new_found = True
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[demo] Skipping unreadable image: {img_path}")
                    processed.add(img_path.name)
                    continue
                o = self.process(img)
                if o is None:
                    print(f"[demo] No results for {img_path.name}")
                elif o.H is None:
                    print(f"[demo] Board not detected in {img_path.name}; mapping skipped.")
                    self.emit(out_dir, session_id, img_path.name, o)
                else:
                    self.emit(out_dir, session_id, img_path.name, o)
                processed.add(img_path.name)
            if not new_found:
                time.sleep(0.5)

    def close(self) -> None:
        if self.client:
            self.client.close()


def main():
    p = argparse.ArgumentParser(description="Planet-C demo runner")
    p.add_argument("--frames-dir", type=str, default=None, help="Frames directory (default: latest phone-server session)")
    p.add_argument("--weights", type=str, default=None, help="YOLO weights path (default: newest under 03-training)")
    # Planet-C grid: 5 rows (A-E) x 4 cols (1-4); cell 1 = A1 = inner-board top-left (order_quad anchors TL).
    p.add_argument("--rows", type=int, default=5, help="Board rows A-E (default 5)")
    p.add_argument("--cols", type=int, default=4, help="Board cols 1-4 (default 4)")
    p.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    p.add_argument("--ws-url", type=str, default=None, help="WebSocket url for CORMAS (e.g. ws://localhost:8081/ws)")
    p.add_argument("--json-out", type=str, default=None, help="Directory for per-session board-state JSON (CORMAS file bridge)")
    p.add_argument("--once", action="store_true", help="Process the folder once and exit (no watch loop)")
    args = p.parse_args()

    frames_dir = find_latest_frames_dir(Path(args.frames_dir) if args.frames_dir else None)
    weights_path = resolve_weights_path(Path(args.weights) if args.weights else None)
    session_id = frames_dir.name if frames_dir.name != "frames" else "session"
    out_dir = PROJECT_ROOT / "frames" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo] Frames directory: {frames_dir}")
    print(f"[demo] Weights: {weights_path}")
    print(f"[demo] Grid: {args.rows}x{args.cols} (cell 1 = A1, top-left)")

    pipe = Pipeline(weights_path, rows=args.rows, cols=args.cols, conf=args.conf, iou=args.iou,
                    json_out=args.json_out, ws_url=args.ws_url)
    try:
        if args.once:
            pipe.run_once(frames_dir, out_dir, session_id)
        else:
            pipe.watch(frames_dir, out_dir, session_id)
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
