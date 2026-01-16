import os
from fastapi import UploadFile

def save_upload_to_disk(upload: UploadFile, upload_dir: str, filename: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, filename)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path
