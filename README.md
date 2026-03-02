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


