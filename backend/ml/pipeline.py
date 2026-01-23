"""Unified ML pipeline for product integration."""

from __future__ import annotations
import os
from typing import Dict, List
from app.schemas import AnalyzeResponse, DefectLocalization, Region

from ml.stage1_binary import predict_stage1
from ml.stage2_classifier import predict_stage2
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
    # --- Stage 1: Binary Classification ---
    s1 = predict_stage1(image_path)

    # --- Stage 2: YOLO Object Detection ---
    s2 = predict_stage2(image_path)

    # --- Stage 3: Localization (Spatial Logic) ---
    yolo_localizations: List[DefectLocalization] = []
    
    print(f"\n--- Pipeline Inference: {image_id} ---")
    
    detections = s2.debug.get("detections", [])
    for det in detections:
        label = det.get("label", "Unknown")
        confidence = det.get("confidence", 0.0)
        location = det.get("location", "N/A")
        box = det.get("box_2d", [0, 0, 0, 0])

        print(f"DEFECT: {label:10} | CONF: {confidence:.2f} | LOC: {location:12} | BOX: {box}")

        region = Region(
            x1=int(box[0]),
            y1=int(box[1]),
            x2=int(box[2]),
            y2=int(box[3]),
            score=confidence
        )
        
        yolo_localizations.append(DefectLocalization(
            label=label,
            confidence=confidence,
            method=f"yolo ({location})", 
            regions=[region]
        ))

    # --- Stage 4: Geometry (Pillow alignment) ---
    s4 = stage4_alignment(image_path, artifact_dir=artifact_dir, base_url=base_url)

    # --- Artifacts & Visuals ---
    artifacts: Dict[str, str] = {}
    
    # FIX: Correctly construct the URL to avoid "static/static"
    annotated_path = s2.debug.get("annotated_image_path") # e.g. 'static/out_xxx.jpg'
    if annotated_path:
        # This will now result in: http://localhost:8000/static/out_xxx.jpg
        # Which matches your 'app.mount("/static", ...)' exactly.
        artifacts["YOLO Detections"] = f"{base_url}/{annotated_path}"
        
    artifacts.update(s4.artifacts)

    # --- Summary Metrics ---
    defects = {
        "Items": s2.probs.get("Items", 0.0),
        "Untucked": s2.probs.get("Untucked", 0.0),
        "Wrinkles": s2.probs.get("Wrinkles", 0.0)
    }

    print(f"STAGE 1 Made Probability: {s1.prob_made:.2f}")
    print(f"STAGE 4 Alignment Score: {s4.alignment_score:.2f}")
    print("-------------------------------------------\n")

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
        localizations=yolo_localizations,
        alignment_score=s4.alignment_score,
        alignment_pass=s4.alignment_pass,
        alignment_debug=s4.debug,
        artifacts=artifacts,
        stage5=robustness,
    )