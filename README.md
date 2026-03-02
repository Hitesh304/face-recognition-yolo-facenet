# Face Recognition System using YOLO + FaceNet

##  Project Overview

This project implements a face recognition pipeline using:

- **YOLO (You Only Look Once)** for face detection
- **FaceNet** for generating facial embeddings
- **Cosine Similarity** for identity matching

The system detects faces in a group image and identifies the person by comparing embeddings with a pre-built database.

---

##  Pipeline Architecture

1. Detect faces in image using YOLO
2. Crop detected faces
3. Resize to 160x160
4. Generate 128D facial embeddings using FaceNet
5. L2 normalize embeddings
6. Compare with stored database embeddings using cosine similarity
7. Display best match above threshold

---

##  Why YOLO + FaceNet?

- YOLO provides fast and accurate face detection.
- FaceNet generates discriminative embeddings.
- One-shot learning approach — no retraining required when adding new identities.

---

##  Project Structure


face-recognition-yolo-facenet/
│
├── src/
│ ├── build_database.py
│ ├── recognize.py
│
├── models/
├── input_images/
│ ├── Db_buildup_images/
│ └── Fresh_group_images/
│
├── Db_embeddings.txt
├── requirements.txt
└── README.md

##  Installation

Clone the repository:

```bash
git clone https://github.com/Hitesh304/face-recognition-yolo-facenet.git
cd face-recognition-yolo-facenet
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
#install the dependencies in virtual enviroment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

Build Database

Place labeled face images inside:

input_images/Db_buildup_images/

Then run:

python src/build_database.py

This generates Db_embeddings.txt.

🔍 Run Face Recognition

Place group image inside:

input_images/Fresh_group_images/

Update image name in recognize.py if needed.

Then run:

python src/recognize.py

The system will:

Detect faces

Compare embeddings

Display best matched identity

Similarity Threshold

Cosine similarity threshold used: 0.6

Value chosen experimentally

Can be tuned based on dataset

🛠 Technologies Used

Python

Ultralytics YOLO

TensorFlow

Keras-FaceNet

OpenCV

NumPy

