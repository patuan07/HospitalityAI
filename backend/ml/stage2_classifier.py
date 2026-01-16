"""Stage 2: Multi-label defect classification.

Input: a bed image
Output: probabilities for multiple defect labels (e.g., wrinkles, stain,
pillow_misaligned).

This template provides a *runnable* heuristic baseline so the product demo
works without model weights.

Teams should replace `predict_stage2()` with their trained model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np


# Keep labels stable; judges/leaderboard can rely on these keys.
DEFAULT_LABELS: List[str] = [
    "wrinkles",
    "stain",
    "pillow_misaligned",
    "blanket_off_center",
]


@dataclass
class Stage2Result:
    probs: Dict[str, float]
    debug: Dict


def _read_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def predict_stage2(
    image_path: str,
    *,
    labels: List[str] | None = None,
    input_width: int = 640,
) -> Stage2Result:
    """Return multi-label defect probabilities.

    Baseline heuristic (for demo only):
    - wrinkles: edge density
    - stain: dark-blob ratio (very rough)
    - pillow_misaligned: defer to Stage-4 (geometry), return placeholder
    - blanket_off_center: left-right intensity imbalance

    Replace this with your trained Stage-2 model.
    """

    labels = labels or DEFAULT_LABELS
    img = _read_bgr(image_path)
    h, w = img.shape[:2]
    if w != input_width:
        scale = input_width / float(w)
        img = cv2.resize(img, (input_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 50, 150)
    edge_density = float(np.mean(edges > 0))
    wrinkles = float(np.clip(edge_density / 0.18, 0.0, 1.0))

    # stain: threshold very dark pixels after normalization
    norm = cv2.normalize(gray_blur, None, 0, 255, cv2.NORM_MINMAX)
    dark_ratio = float(np.mean(norm < 40))
    stain = float(np.clip(dark_ratio / 0.08, 0.0, 1.0))

    # blanket_off_center: compare left vs right mean intensity
    mid = norm.shape[1] // 2
    left_mean = float(np.mean(norm[:, :mid]))
    right_mean = float(np.mean(norm[:, mid:]))
    imbalance = abs(left_mean - right_mean) / 255.0
    blanket_off_center = float(np.clip(imbalance / 0.12, 0.0, 1.0))

    # pillow_misaligned: placeholder here (Stage-4 is the real metric)
    pillow_misaligned = 0.0

    probs_map = {
        "wrinkles": wrinkles,
        "stain": stain,
        "pillow_misaligned": pillow_misaligned,
        "blanket_off_center": blanket_off_center,
    }

    # Return only requested labels
    probs = {k: float(probs_map.get(k, 0.0)) for k in labels}

    return Stage2Result(
        probs=probs,
        debug={
            "edge_density": edge_density,
            "dark_ratio": dark_ratio,
            "left_mean": left_mean,
            "right_mean": right_mean,
            "note": "heuristic baseline; replace with your trained Stage-2 model",
        },
    )
