from fastapi import FastAPI, File, UploadFile
from typing import List
import shutil
from pathlib import Path

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

#image  upload api for building database, not for recognition

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR.parent / "input_images" / "Db_buildup_images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/register")
async def upload_multiple_images(files: List[UploadFile] = File(...)):
    uploaded_files = []

    for file in files:
        file_location = UPLOAD_DIR / file.filename

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        size = file.file.seek(0, 2)
        file.file.seek(0)

        uploaded_files.append({
            "filename": file.filename,
            "size": size
        })

    return {"uploaded_files": uploaded_files}

#recognize api
from fastapi.responses import FileResponse

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    file_location = BASE_DIR.parent / "input_images" / file.filename

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Call recognition logic
    from recognize import generate_attendance
    csv_path = generate_attendance()   # IMPORTANT: ye CSV ka path return kare

    return FileResponse(
        path=csv_path,
        media_type='text/csv',
        filename="attendance.csv"
    )