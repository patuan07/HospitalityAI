"""Unified ML pipeline for product integration.

Stages
------
Stage 1: binary classification
Stage 2: multi-label classification
Stage 3: localization (weak CAM)
Stage 4: geometry-based pillow alignment
Stage 5: robustness evaluation (optional)

This pipeline returns a single Pydantic response (AnalyzeResponse)
that the web/mobile UI can consume directly.
"""

from __future__ import annotations

from typing import Dict, List

from app.schemas import AnalyzeResponse

from ml.stage1_binary import predict_stage1
from ml.stage2_classifier import predict_stage2
from ml.stage3_weak_cam import run_stage3_weak_cam
from ml.stage4_geometry import stage4_alignment
from ml.stage5_robustness import evaluate_robustness


def run_pipeline(
    *,
    image_path: str,
    image_id: str,
    artifact_dir: str,
    base_url: str,
    run_stage5: bool = False,
) -> AnalyzeResponse:
    # Stage 1
    s1 = predict_stage1(image_path)

    # Stage 2
    s2 = predict_stage2(image_path)

    # Stage 3 (localization)
    s3 = run_stage3_weak_cam(
        image_path=image_path,
        defects=s2.probs,
        artifact_dir=artifact_dir,
        base_url=base_url,
        topk=2,
        threshold=0.5,
    )

    # Stage 4 (geometry)
    s4 = stage4_alignment(image_path, artifact_dir=artifact_dir, base_url=base_url)

    # Optional: reflect Stage4 into Stage2 pillow_misaligned probability for the demo
    # (so the UI shows a coherent story)
    defects = dict(s2.probs)
    defects["pillow_misaligned"] = float(1.0 - s4.alignment_score)

    artifacts: Dict[str, str] = {}
    artifacts.update(s3.artifacts)
    artifacts.update(s4.artifacts)

    stages_run: List[str] = ["stage1", "stage2", "stage3", "stage4"]
    robustness = None
    if run_stage5:
        s5 = evaluate_robustness(image_path, artifact_dir=artifact_dir, base_url=base_url)
        robustness = {
            "robustness_score": s5.robustness_score,
            "details": s5.details,
        }
        stages_run.append("stage5")

    return AnalyzeResponse(
        image_id=image_id,
        stages_run=stages_run,
        stage1={"prob_made": s1.prob_made, "pred_made": s1.pred_made, "debug": s1.debug},
        defects=defects,
        localizations=s3.localizations,
        alignment_score=s4.alignment_score,
        alignment_pass=s4.alignment_pass,
        alignment_debug=s4.debug,
        artifacts=artifacts,
        stage5=robustness,
    )
