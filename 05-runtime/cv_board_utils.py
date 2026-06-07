import cv2
import numpy as np
from typing import Optional, Tuple


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_board_edges_and_quad(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return edges, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            return edges, _order_points(quad)
    return edges, None


def detect_board_quad(img: np.ndarray) -> Optional[np.ndarray]:
    edges, quad = detect_board_edges_and_quad(img)
    return quad


def compute_homography(quad: np.ndarray) -> np.ndarray:
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return H


def transform_point(H: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    src = np.array([[pt]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)[0][0]
    x = float(np.clip(dst[0], 0.0, 1.0))
    y = float(np.clip(dst[1], 0.0, 1.0))
    return x, y


def cell_index(x_norm: float, y_norm: float, rows: int, cols: int) -> int:
    r = int(np.floor(y_norm * rows))
    c = int(np.floor(x_norm * cols))
    r = max(0, min(rows - 1, r))
    c = max(0, min(cols - 1, c))
    return r * cols + c + 1