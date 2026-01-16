"""Pydantic schemas (API contract).

This is the *single source of truth* for what the product UI expects.
Teams may change internal ML code, but should keep this response contract
stable for integration + judging.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Region(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class DefectLocalization(BaseModel):
    label: str
    confidence: float
    method: str  # e.g. "gradcam", "cam_fallback", "geometry", "yolo"
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    regions: Optional[List[Region]] = None


class AnalyzeResponse(BaseModel):
    image_id: str
    stages_run: List[str]

    # Stage 1 (binary)
    stage1: Dict[str, Any]

    # Stage 2 (multi-label): label -> probability
    defects: Dict[str, float]

    # Stage 3 (localization)
    localizations: List[DefectLocalization]

    # Stage 4 (geometry alignment)
    alignment_score: float
    alignment_pass: bool
    alignment_debug: Dict[str, Any] = {}

    # Stage 5 (robustness) optional
    stage5: Optional[Dict[str, Any]] = None

    # UI-friendly artifact URLs
    artifacts: Dict[str, str] = {}
