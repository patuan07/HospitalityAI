import os
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

from app.schemas import AnalyzeResponse
from app.settings import settings
from app.utils_images import save_upload_to_disk
from ml.pipeline import run_pipeline


app = FastAPI(title="Hospitality AI Product Template")


os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)

# Serve generated images (overlays/heatmaps/rotated views) for UI display
app.mount("/artifacts", StaticFiles(directory=settings.ARTIFACT_DIR), name="artifacts")


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    run_stage5: bool = False,
) -> AnalyzeResponse:
    """Analyze a single image and return Stage1..Stage4 (and optional Stage5).

    `run_stage5=true` enables the robustness evaluator.
    """

    image_id = f"{uuid.uuid4().hex}_{image.filename}"
    image_path = save_upload_to_disk(image, settings.UPLOAD_DIR, image_id)

    return run_pipeline(
        image_path=image_path,
        image_id=image_id,
        artifact_dir=settings.ARTIFACT_DIR,
        base_url=settings.ARTIFACT_BASE_URL,
        run_stage5=run_stage5,
    )
