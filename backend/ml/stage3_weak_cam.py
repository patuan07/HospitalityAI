"""Stage 3: Localization (weak CAM).

Goal
----
Produce a heatmap/overlay that shows *where* the model looked for a given
defect label.

This module is designed to work in two modes:

1) Lightweight fallback (default):
   - No torch required
   - Heatmap is derived from edges + local contrast (a "visual" baseline)

2) Real Grad-CAM mode (optional):
   - If teams install torch + pytorch-grad-cam and provide a model,
     they can replace `compute_cam_heatmap()` with true Grad-CAM.

For the competition template, we prioritize *product integration*:
students must return overlays/heatmaps and connect them to a web/mobile UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import os
import cv2
import numpy as np

from app.schemas import DefectLocalization


@dataclass
class Stage3Result:
    localizations: List[DefectLocalization]
    artifacts: Dict[str, str]
    debug: Dict


def _read_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def _write_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _to_url(base_url: str, artifact_path: str, artifact_dir: str) -> str:
    rel = os.path.relpath(artifact_path, artifact_dir).replace("\\", "/")
    return f"{base_url.rstrip('/')}/{rel}"


def compute_cam_heatmap_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """Fallback "CAM": edges + contrast.

    Returns a float heatmap in [0,1] with same HxW.
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

    # Local contrast (Laplacian magnitude)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap = np.abs(lap)
    lap = cv2.normalize(lap, None, 0.0, 1.0, cv2.NORM_MINMAX)

    heatmap = 0.6 * edges + 0.4 * lap
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), 3)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap


def overlay_heatmap(img_bgr: np.ndarray, heatmap01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_u8 = np.uint8(np.clip(heatmap01 * 255.0, 0, 255))
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)
    return out


def run_stage3_weak_cam(
    *,
    image_path: str,
    defects: Dict[str, float],
    artifact_dir: str,
    base_url: str,
    topk: int = 2,
    threshold: float = 0.5,
) -> Stage3Result:
    """Generate CAM overlays for the top defects.

    By default, we choose top-k labels whose probability >= threshold.
    """

    img = _read_bgr(image_path)

    # Choose candidate labels
    ordered = sorted(defects.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [k for k, v in ordered if v >= threshold][:topk]

    localizations: List[DefectLocalization] = []
    artifacts: Dict[str, str] = {}

    for label in chosen:
        heatmap = compute_cam_heatmap_fallback(img)
        overlay = overlay_heatmap(img, heatmap)

        heat_path = os.path.join(artifact_dir, f"stage3_{label}_heatmap.png")
        ov_path = os.path.join(artifact_dir, f"stage3_{label}_overlay.png")
        _write_image(heat_path, np.uint8(heatmap * 255.0))
        _write_image(ov_path, overlay)

        heat_url = _to_url(base_url, heat_path, artifact_dir)
        ov_url = _to_url(base_url, ov_path, artifact_dir)

        localizations.append(
            DefectLocalization(
                label=label,
                confidence=float(defects[label]),
                method="cam_fallback",
                heatmap_path=heat_url,
                overlay_path=ov_url,
                regions=None,
            )
        )

        artifacts[f"stage3_{label}_heatmap"] = heat_url
        artifacts[f"stage3_{label}_overlay"] = ov_url

    return Stage3Result(
        localizations=localizations,
        artifacts=artifacts,
        debug={
            "topk": topk,
            "threshold": threshold,
            "chosen": chosen,
            "note": "fallback CAM; replace with true Grad-CAM if desired",
        },
    )
