from fastapi import FastAPI, File, UploadFile
from typing import List
import shutil
from pathlib import Path
from fastapi.responses import HTMLResponse, FileResponse
import os

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "input_images" / "Db_buildup_images"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

RECOGNIZE_DIR = BASE_DIR / "input_images"
RECOGNIZE_DIR.mkdir(parents=True, exist_ok=True)

# ------------------ UI ------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Face Recognition</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                text-align: center;
                margin: 0;
                padding: 0;
            }

            .container {
                margin-top: 50px;
            }

            h1 {
                font-size: 40px;
                margin-bottom: 30px;
            }

            .card {
                background: white;
                color: black;
                padding: 20px;
                margin: 20px auto;
                width: 320px;
                border-radius: 15px;
                box-shadow: 0px 5px 20px rgba(0,0,0,0.3);
            }

            input[type="file"] {
                margin: 10px 0;
            }

            button {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                background: #667eea;
                color: white;
                cursor: pointer;
                font-size: 16px;
            }

            button:hover {
                background: #5a67d8;
            }

            .loader {
                display: none;
                margin-top: 20px;
            }

            .spinner {
                border: 6px solid #f3f3f3;
                border-top: 6px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>

        <script>
            function showLoader() {
                document.getElementById("loader").style.display = "block";
            }
        </script>
    </head>

    <body>
        <div class="container">
            <h1>🚀 Face Recognition System</h1>

            <div class="card">
                <h2>Register Faces</h2>
                <form action="/register" method="post" enctype="multipart/form-data" onsubmit="showLoader()">
                    <input type="file" name="filess" multiple required><br>
                    <button type="submit">Register</button>
                </form>
            </div>

            <div class="card">
                <h2>Recognize Face</h2>
                <form action="/recognize" method="post" enctype="multipart/form-data" onsubmit="showLoader()">
                    <input type="file" name="file" required><br>
                    <button type="submit">Recognize</button>
                </form>
            </div>

            <div id="loader" class="loader">
                <div class="spinner"></div>
                <p>Processing... please wait</p>
            </div>
        </div>
    </body>
    </html>
    """

# ------------------ REGISTER API ------------------
from build_database import build_database
@app.post("/register")
async def upload_multiple_images(filess: List[UploadFile] = File(...)):
    uploaded_files = []

    for file in filess:
        file_location = UPLOAD_DIR / file.filename

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded_files.append(file.filename)
    build_database()

    return {
        "status": "success",
        "uploaded": uploaded_files
    }

# ------------------ RECOGNIZE API ------------------

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    file_location = RECOGNIZE_DIR / file.filename

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 Tumhara actual logic yaha call hoga
    from recognize import generate_attendance
    csv_path = generate_attendance()

    return FileResponse(
        path=csv_path,
        media_type='text/csv',
        filename="attendance.csv"
    )

# ------------------ RUN ------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)