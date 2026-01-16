"""Stage 1: Binary classification baseline.

Task example: "Is the bed made?" -> {0,1}

This template intentionally works without torch. The default implementation
is a lightweight heuristic (edge density) to keep the product template
fully runnable.

Teams should replace `predict_stage1()` with their trained model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


@dataclass
class Stage1Result:
    prob_made: float
    pred_made: bool
    debug: Dict


def _read_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def predict_stage1(
    image_path: str,
    *,
    input_width: int = 640,
    edge_threshold: float = 0.085,
) -> Stage1Result:
    """Return probability that the bed is made.

    Baseline heuristic idea:
    - A made bed tends to have cleaner, larger contiguous surfaces
    - An unmade bed tends to have more edges (wrinkles, clutter)

    We approximate this with an edge-density score.
    """

    img = _read_bgr(image_path)

    # Resize preserving aspect ratio
    h, w = img.shape[:2]
    if w != input_width:
        scale = input_width / float(w)
        img = cv2.resize(img, (input_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = float(np.mean(edges > 0))  # 0..1

    # Heuristic: lower edge density => more likely "made"
    # Map density to probability with a simple clamp.
    prob_made = float(np.clip((edge_threshold - edge_density) / edge_threshold, 0.0, 1.0))
    pred_made = prob_made >= 0.5

    return Stage1Result(
        prob_made=prob_made,
        pred_made=pred_made,
        debug={
            "edge_density": edge_density,
            "edge_threshold": edge_threshold,
            "note": "heuristic baseline; replace with your trained Stage-1 model",
        },
    )
