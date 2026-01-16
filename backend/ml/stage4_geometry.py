"""Stage 4: Geometry-based pillow/bed alignment.

Baseline (defensible geometry):
1) Resize + grayscale + blur
2) Canny edges
3) HoughLinesP to find dominant bed edges
4) Estimate dominant bed axis angle
5) Rotate image to canonical orientation
6) Compute left-right edge symmetry around the bed centerline

Returns:
  - alignment_score in [0,1] (higher is better)
  - alignment_pass (thresholded)
  - debug dict
  - artifact URLs for UI display
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import os
import cv2
import numpy as np


@dataclass
class Stage4Result:
    alignment_score: float
    alignment_pass: bool
    debug: Dict
    artifacts: Dict[str, str]


def _write_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _to_url(base_url: str, artifact_path: str, artifact_dir: str) -> str:
    rel = os.path.relpath(artifact_path, artifact_dir).replace("\\", "/")
    return f"{base_url.rstrip('/')}/{rel}"


def _rotate(img: np.ndarray, angle_deg: float, *, is_mask: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_val = 0
    return cv2.warpAffine(img, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)


def _dominant_angle_hist(angles_deg: np.ndarray, weights: np.ndarray) -> float:
    """Find dominant angle in [-90,90] using weighted histogram peak."""
    # bins for -90..90 inclusive
    bins = np.linspace(-90.0, 90.0, 181)
    hist = np.zeros(len(bins) - 1, dtype=np.float64)
    inds = np.clip(np.digitize(angles_deg, bins) - 1, 0, len(hist) - 1)
    for i, w in zip(inds, weights):
        hist[i] += float(w)

    peak = int(np.argmax(hist))
    # average within +-5 degrees around peak
    low = max(0, peak - 5)
    high = min(len(hist) - 1, peak + 5)
    mask = (inds >= low) & (inds <= high)
    if not np.any(mask):
        return float(angles_deg[np.argmax(weights)])

    return float(np.average(angles_deg[mask], weights=weights[mask]))


def stage4_alignment(
    image_path: str,
    *,
    artifact_dir: str,
    base_url: str,
    input_width: int = 640,
    symmetry_threshold: float = 0.75,
    roi: Tuple[float, float, float, float] = (0.15, 0.25, 0.85, 0.95),
) -> Stage4Result:
    """Compute geometry alignment score and generate debug artifacts."""

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # resize
    h, w = img.shape[:2]
    if w != input_width:
        scale = input_width / float(w)
        img = cv2.resize(img, (input_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(0.15 * img.shape[1]),
        maxLineGap=20,
    )

    if lines is None:
        score = 0.0
        passed = False
        debug = {"reason": "no_hough_lines"}
        return Stage4Result(score, passed, debug, artifacts={})

    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 0.15 * img.shape[1]:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))  # -180..180
        # map to [-90,90]
        if ang < -90:
            ang += 180
        if ang > 90:
            ang -= 180
        angles.append(ang)
        weights.append(length)

    if not angles:
        score = 0.0
        passed = False
        debug = {"reason": "no_valid_lines"}
        return Stage4Result(score, passed, debug, artifacts={})

    angles_arr = np.array(angles, dtype=np.float32)
    weights_arr = np.array(weights, dtype=np.float32)
    dominant_angle = _dominant_angle_hist(angles_arr, weights_arr)

    rot_img = _rotate(img, -dominant_angle)
    rot_edges = _rotate(edges, -dominant_angle, is_mask=True)

    H, W = rot_edges.shape[:2]
    x1 = int(roi[0] * W)
    y1 = int(roi[1] * H)
    x2 = int(roi[2] * W)
    y2 = int(roi[3] * H)
    roi_edges = rot_edges[y1:y2, x1:x2]

    # symmetry around ROI centerline
    hh, ww = roi_edges.shape[:2]
    mid = ww // 2
    left = roi_edges[:, :mid]
    right = roi_edges[:, mid : mid + left.shape[1]]
    right_flip = cv2.flip(right, 1)

    L = (left > 0).astype(np.float32)
    R = (right_flip > 0).astype(np.float32)
    mad = float(np.mean(np.abs(L - R)))
    score = float(np.clip(1.0 - mad, 0.0, 1.0))
    passed = score >= symmetry_threshold

    # --- artifacts
    artifacts: Dict[str, str] = {}
    rot_path = os.path.join(artifact_dir, "stage4_rotated.png")
    edges_path = os.path.join(artifact_dir, "stage4_edges_rotated.png")
    overlay_path = os.path.join(artifact_dir, "stage4_roi_overlay.png")

    # edges visualization
    edges_vis = cv2.cvtColor(rot_edges, cv2.COLOR_GRAY2BGR)
    _write_image(rot_path, rot_img)
    _write_image(edges_path, edges_vis)

    overlay = rot_img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.line(overlay, (x1 + (x2 - x1) // 2, y1), (x1 + (x2 - x1) // 2, y2), (0, 255, 255), 2)
    _write_image(overlay_path, overlay)

    artifacts["stage4_rotated"] = _to_url(base_url, rot_path, artifact_dir)
    artifacts["stage4_edges"] = _to_url(base_url, edges_path, artifact_dir)
    artifacts["stage4_roi_overlay"] = _to_url(base_url, overlay_path, artifact_dir)

    debug = {
        "dominant_angle_deg": float(dominant_angle),
        "roi_px": [x1, y1, x2, y2],
        "mad": mad,
        "symmetry_threshold": symmetry_threshold,
    }

    return Stage4Result(score, passed, debug, artifacts)
