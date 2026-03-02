import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from keras_facenet import FaceNet
from pathlib import Path

# ================= PATH CONFIG (PORTABLE) =================
BASE_DIR = Path(__file__).resolve().parent.parent

YOLO_MODEL_PATH = BASE_DIR / "models" / "Face_detection_trained_for_10epochs.pt"
INPUT_FOLDER = BASE_DIR / "input_images" / "Db_buildup_images"
DB_FILE = BASE_DIR / "Db_embeddings.txt"

YOLO_CONF_THRESHOLD = 0.75
FACENET_IMAGE_SIZE = 160
# ==========================================================


# -------- LOAD MODELS --------
yolo_model = YOLO(str(YOLO_MODEL_PATH))
facenet = FaceNet()


# -------- FACE DETECTION --------
def detect_and_crop(image_path):

    results = yolo_model(str(image_path), conf=YOLO_CONF_THRESHOLD, verbose=False)

    if len(results[0].boxes) == 0:
        print(f"No face detected in {image_path}")
        return None

    boxes = results[0].boxes
    best_idx = boxes.conf.argmax()

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_jpeg(img, channels=3)

    face = img[y1:y2, x1:x2]

    return face


# -------- PREPROCESS --------
def preprocess(face_img):
    face_img = tf.image.resize(face_img, (FACENET_IMAGE_SIZE, FACENET_IMAGE_SIZE))
    face_img = tf.cast(face_img, tf.float32)
    return face_img.numpy()


# -------- BUILD DATABASE --------
def build_database():

    if not INPUT_FOLDER.exists():
        print("Input folder not found.")
        return

    print("Building database...\n")

    with open(DB_FILE, "w") as f:

        for img_path in INPUT_FOLDER.iterdir():

            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            person_name = img_path.stem
            print(f"Processing {person_name}...")

            face = detect_and_crop(img_path)

            if face is None:
                continue

            face = preprocess(face)

            embedding = facenet.embeddings([face])[0]

            # L2 normalize
            embedding = embedding / np.linalg.norm(embedding)

            embedding_str = ",".join(map(str, embedding))
            f.write(f"{person_name}:{embedding_str}\n")

            print(f"Stored embedding for {person_name}\n")

    print("Database building complete.")


if __name__ == "__main__":
    build_database()