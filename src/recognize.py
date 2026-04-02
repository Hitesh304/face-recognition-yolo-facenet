# csv format
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import csv
from ultralytics import YOLO
from keras_facenet import FaceNet
import pathlib

# Windows compatibility fix
pathlib.PosixPath = pathlib.WindowsPath


BASE_DIR = Path(__file__).parent.parent

# ================= PARAMETERS =================
YOLO_MODEL_PATH = BASE_DIR / "models" / "Face_detection_trained_for_10epochs.pt"
INPUT_DIR = BASE_DIR / "input_images"
DB_FILE = BASE_DIR / "Db_embeddings.txt"
OUTPUT_CSV = BASE_DIR / "attendance.csv"

YOLO_CONF_THRESHOLD = 0.4
THRESHOLD = 0.65
IMAGE_SIZE = 160
# ==============================================


# -------- GET IMAGE DYNAMICALLY --------
def get_group_image_path():
    extensions = [".jpg", ".jpeg", ".png"]

    images = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in extensions]

    if not images:
        raise ValueError("❌ No image found in input_images folder")

    # pick latest image (best practice)
    images = sorted(images, key=lambda x: x.stat().st_mtime, reverse=True)

    if len(images) > 1:
        print(f"⚠️ Multiple images found, using latest: {images[0].name}")

    return images[0]


# -------- LOAD MODELS --------
yolo_model = YOLO(YOLO_MODEL_PATH)
facenet = FaceNet()


# -------- LOAD DATABASE --------
def load_database():
    db_embeddings = []
    db_names = []

    with open(DB_FILE, "r") as f:
        for line in f:
            name, values = line.strip().split(":")
            embedding = np.array(list(map(float, values.split(","))))
            embedding = embedding / np.linalg.norm(embedding)
            db_names.append(name)
            db_embeddings.append(embedding)

    return db_names, db_embeddings


# -------- COSINE SIMILARITY --------
def cosine_similarity(a, b):
    return np.dot(a, b)


# -------- MAIN FUNCTION --------
def generate_attendance():

    GROUP_IMAGE_PATH = get_group_image_path()
    print(f"📸 Using image: {GROUP_IMAGE_PATH.name}")

    db_names, db_embeddings = load_database()

    attendance = {name: "Absent" for name in db_names}

    results = yolo_model(
        GROUP_IMAGE_PATH,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:
        print("❌ No faces detected.")
    else:
        img = cv2.imread(str(GROUP_IMAGE_PATH))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_img.shape

        matched_db_indices = set()

        for box in boxes.xyxy:

            x1, y1, x2, y2 = box.cpu().numpy().astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = rgb_img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            embedding = facenet.embeddings([face_resized])[0]
            embedding = embedding / np.linalg.norm(embedding)

            best_score = -1
            best_index = -1

            for i, db_embedding in enumerate(db_embeddings):

                if i in matched_db_indices:
                    continue

                score = cosine_similarity(embedding, db_embedding)

                if score > best_score:
                    best_score = score
                    best_index = i

            if best_score >= THRESHOLD:
                attendance[db_names[best_index]] = "Present"
                matched_db_indices.add(best_index)

    # -------- SORT ATTENDANCE --------
    sorted_attendance = sorted(
        attendance.items(),
        key=lambda x: x[1] == "Absent"
    )

    # -------- WRITE CSV --------
    with open(OUTPUT_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Attendance"])
        writer.writerows(sorted_attendance)

    print("✅ Attendance CSV Generated:", OUTPUT_CSV)
    return OUTPUT_CSV


# -------- RUN --------
if __name__ == "__main__":
    generate_attendance()