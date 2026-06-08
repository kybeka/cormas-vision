"""
Derive the board quad from the YOLO model's own `inner-board` / `board` OBB
detection instead of Canny edges.

The model detects `inner-board` as an oriented box at high confidence and is far
steadier frame-to-frame than the largest-contour heuristic in cv_board_utils.py.
Using the inner-board corners also aligns the homography to the actual 5x4 grid
(Canny tends to grab the whole mat). A small smoother absorbs residual wobble
since the board is static during play.
"""
from __future__ import annotations
from typing import Optional, Sequence
import numpy as np


def order_quad(pts: Sequence) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (row-major board space)."""
    p = np.asarray(pts, np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()  # x - y
    return np.array([p[np.argmin(s)],   # TL  (min x+y)
                     p[np.argmin(d)],    # TR  (min x-y -> wait: max? handled below)
                     p[np.argmax(s)],    # BR  (max x+y)
                     p[np.argmax(d)]],   # BL
                    np.float32)


def board_quad_from_result(result, prefer=("inner-board", "board")) -> Optional[np.ndarray]:
    """Return the ordered 4-corner board quad (pixels) from a YOLO OBB result, or None."""
    obb = getattr(result, "obb", None)
    if obb is None or getattr(obb, "xyxyxyxy", None) is None or len(obb) == 0:
        return None
    names = getattr(result, "names", {})
    polys = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    cls = obb.cls.cpu().numpy().astype(int)
    conf = obb.conf.cpu().numpy()
    for want in prefer:
        idxs = [i for i, c in enumerate(cls) if names.get(c, "") == want]
        if idxs:
            best = max(idxs, key=lambda i: conf[i])
            return order_quad(polys[best])
    return None


class QuadSmoother:
    """Exponential moving average of an ordered quad across frames."""
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.q: Optional[np.ndarray] = None

    def update(self, quad: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if quad is None:
            return self.q
        self.q = quad.astype(np.float32) if self.q is None \
            else self.alpha * quad + (1 - self.alpha) * self.q
        return self.q
