import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

# 1. Initialize the app
app = FastAPI(title="Hospitality AI Product Template")

# 2. Imports from your own modules
from app.schemas import AnalyzeResponse
from app.settings import settings
from app.utils_images import save_upload_to_disk
from ml.pipeline import run_pipeline

# 3. Setup directories and mounts
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACT_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/artifacts", StaticFiles(directory=settings.ARTIFACT_DIR), name="artifacts")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
async def health() -> dict:
    return {"ok": True}

# 4. Corrected Analyze Endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...), run_stage5: bool = False):
    # --- IMAGE SAVING LOGIC (The missing part) ---
    # Create a unique ID for the image to prevent filename collisions
    image_id = f"{uuid.uuid4().hex}_{image.filename}"
    
    # Save the uploaded file to your UPLOAD_DIR and get the full path
    image_path = save_upload_to_disk(image, settings.UPLOAD_DIR, image_id)

    # Base URL for artifacts and static files
    # This allows the pipeline to build URLs like http://localhost:8000/static/...
    base_url = "http://localhost:8000" 

    # Now 'image_path' and 'image_id' are defined and can be passed safely
    return run_pipeline(
        image_path=image_path,
        image_id=image_id,
        artifact_dir=settings.ARTIFACT_DIR,
        base_url=base_url, 
        run_stage5=run_stage5,
    )