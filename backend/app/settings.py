import os
from pydantic import BaseModel

class Settings(BaseModel):
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    ARTIFACT_DIR: str = os.getenv("ARTIFACT_DIR", "artifacts")
    ARTIFACT_BASE_URL: str = os.getenv("ARTIFACT_BASE_URL", "http://localhost:8000/artifacts")

settings = Settings()
