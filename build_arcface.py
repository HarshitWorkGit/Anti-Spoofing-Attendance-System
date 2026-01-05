#!/usr/bin/env python3
"""
ArcFace + FAISS Training (Scales from 10 → 6000 identities)
CPU & GPU compatible
"""

# ================= USER SWITCH =================
USE_GPU = True   # True → GPU if available | False → CPU only

# ================= ENV SETUP =================
import os
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ================= IMPORTS =================
import time
import numpy as np
import faiss
from deepface import DeepFace
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO
import tensorflow as tf

# ================= GPU CHECK =================
gpus = tf.config.list_physical_devices("GPU")
GPU_ACTIVE = USE_GPU and len(gpus) > 0

if GPU_ACTIVE:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("🧠 Device used : GPU")
else:
    print("🧠 Device used : CPU")

# ================= CONFIG =================
DATASET_PATH = "dataset10"     # change to dataset_6000 later
MODEL_DIR = "models"

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface" if GPU_ACTIVE else "mtcnn"

BATCH_SIZE = 48 if GPU_ACTIVE else 24
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================= LOAD DATA =================
image_paths = []

for person in sorted(os.listdir(DATASET_PATH)):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    for img in os.listdir(person_dir):
        if img.lower().endswith(IMAGE_EXTS):
            image_paths.append((person, os.path.join(person_dir, img)))

total_images = len(image_paths)
identities = len(set(p for p, _ in image_paths))

print("\n📊 Dataset Info")
print(f"👤 Identities : {identities}")
print(f"📸 Images     : {total_images}")
print(f"🚀 Batch size : {BATCH_SIZE}")
print("-" * 50)

# ================= EMBEDDING =================
embeddings = []
labels = []
skipped = 0
start_time = time.time()

num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE

pbar = tqdm(range(num_batches), desc="Embedding (ArcFace)", unit="batch")

for batch_idx in pbar:
    batch = image_paths[
        batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE
    ]

    for person, img_path in batch:
        try:
            with redirect_stdout(StringIO()):
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=True
                )

            emb = np.array(rep[0]["embedding"], dtype="float32")
            emb /= np.linalg.norm(emb)

            embeddings.append(emb)
            labels.append(person)

        except Exception:
            skipped += 1

    elapsed = time.time() - start_time
    speed = len(embeddings) / elapsed if elapsed > 0 else 0
    remaining = total_images - len(embeddings)
    eta_min = (remaining / speed) / 60 if speed > 0 else 0

    pbar.set_postfix({
        "imgs": f"{len(embeddings)}/{total_images}",
        "img/s": f"{speed:.2f}",
        "ETA(min)": f"{eta_min:.1f}"
    })

pbar.close()

# ================= FAISS BUILD =================
print("\n🧠 Building FAISS index...")

embeddings = np.vstack(embeddings).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(MODEL_DIR, "arcface_faiss.index"))
np.save(os.path.join(MODEL_DIR, "labels.npy"), np.array(labels))

# ================= SUMMARY =================
total_time = time.time() - start_time

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print(f"👤 Identities : {identities}")
print(f"🧠 Embeddings : {len(embeddings)}")
print(f"⚠️ Skipped    : {skipped}")
print(f"⏱️ Time      : {total_time/60:.2f} minutes")
print(f"⚡ Speed     : {len(embeddings)/total_time:.2f} img/s")
print("📦 Output:")
print("   - models/arcface_faiss.index")
print("   - models/labels.npy")
print("=" * 60)
