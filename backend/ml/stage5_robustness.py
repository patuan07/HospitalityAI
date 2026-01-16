"""Stage 5: Robustness.

Stage 5 is an *improvement stage*: teams improve robustness of their system
to lighting, noise, blur, small rotations, etc.

In the product template, we provide a simple robustness evaluator:
  - apply a set of deterministic augmentations
  - re-run Stage 1/2/4 baselines
  - report consistency score (0..1)

This is not a leaderboard metric by itself unless you choose to make it one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ml.stage1_binary import predict_stage1
from ml.stage2_classifier import predict_stage2
from ml.stage4_geometry import stage4_alignment


@dataclass
class Stage5Result:
    robustness_score: float
    details: Dict


def _augmentations(img_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Deterministic set of augmentations."""

    outs: List[Tuple[str, np.ndarray]] = [("orig", img_bgr)]

    # brightness up/down
    outs.append(("bright+", cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=25)))
    outs.append(("bright-", cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=-25)))

    # gaussian blur
    outs.append(("blur", cv2.GaussianBlur(img_bgr, (7, 7), 1.2)))

    # gaussian noise
    noise = np.random.default_rng(0).normal(0, 12, img_bgr.shape).astype(np.float32)
    noisy = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    outs.append(("noise", noisy))

    # small rotations
    h, w = img_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    for deg in (-3.0, 3.0):
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        rot = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        outs.append((f"rot{int(deg)}", rot))

    return outs


def evaluate_robustness(
    image_path: str,
    *,
    artifact_dir: str,
    base_url: str,
) -> Stage5Result:
    """Compute a simple robustness score based on consistency.

    Consistency criteria (baseline):
    - Stage1 pred_made stays the same
    - Stage4 alignment_pass stays the same
    - Stage2 top-1 defect label stays the same
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    aug_list = _augmentations(img)

    # Run on original as reference
    ref1 = predict_stage1(image_path)
    ref2 = predict_stage2(image_path)
    ref4 = stage4_alignment(image_path, artifact_dir=artifact_dir, base_url=base_url)

    ref_top = max(ref2.probs.items(), key=lambda kv: kv[1])[0] if ref2.probs else ""

    total = 0
    good = 0
    per_aug: Dict[str, Dict] = {}

    for name, aug in aug_list:
        # write temp to disk? avoid: run by encoding/decoding via memory
        # easiest: save to a temp file in artifact_dir
        tmp_path = f"{artifact_dir}/_tmp_stage5_{name}.png"
        cv2.imwrite(tmp_path, aug)

        s1 = predict_stage1(tmp_path)
        s2 = predict_stage2(tmp_path)
        s4 = stage4_alignment(tmp_path, artifact_dir=artifact_dir, base_url=base_url)

        top = max(s2.probs.items(), key=lambda kv: kv[1])[0] if s2.probs else ""

        ok = (s1.pred_made == ref1.pred_made) and (s4.alignment_pass == ref4.alignment_pass) and (top == ref_top)
        total += 1
        good += 1 if ok else 0

        per_aug[name] = {
            "stage1_pred_made": s1.pred_made,
            "stage2_top": top,
            "stage4_pass": s4.alignment_pass,
            "consistent": ok,
        }

    score = float(good / total) if total else 0.0

    return Stage5Result(
        robustness_score=score,
        details={
            "reference": {
                "stage1_pred_made": ref1.pred_made,
                "stage2_top": ref_top,
                "stage4_pass": ref4.alignment_pass,
            },
            "per_augmentation": per_aug,
            "note": "baseline consistency score; teams can improve Stage5 by making predictions stable",
        },
    )
