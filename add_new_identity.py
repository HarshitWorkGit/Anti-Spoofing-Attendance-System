#!/usr/bin/env python3
"""
Register New Identities Script
- Moves images from 'new_identities_upload' to 'dataset'
- Updates the FAISS index (incremental update)
- Auto-switches CPU/GPU based on workload
"""

import os
import shutil
import sys
import time
import numpy as np
import faiss
import tensorflow as tf
from deepface import DeepFace
from contextlib import redirect_stdout
from io import StringIO

# ================= COLORS =================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

sys.stdout.reconfigure(encoding='utf-8')

# ================= CONFIG =================
UPLOAD_DIR = "Add New Identity"
DATASET_DIR = "dataset"
BATCH_THRESHOLD_FOR_GPU = 50  # If more than 50 images, try GPU

def print_header():
    print(f"\n{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}   🆕 REGISTER NEW IDENTITIES CLI      {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}\n")

def get_model_choice():
    print(f"{Colors.CYAN}Select Model to Update:{Colors.ENDC}")
    print("  [a] ArcFace (Recommended)")
    print("  [f] FaceNet")
    while True:
        m = input(f"{Colors.BOLD}>> Enter choice (a/f): {Colors.ENDC}").strip().lower()
        if m in ['a', 'f']:
            return "ArcFace" if m == 'a' else "FaceNet", "arcface" if m == 'a' else "facenet"
        print(f"{Colors.FAIL}Invalid input!{Colors.ENDC}")

def configure_device(image_count, force_cpu_if_low=True):
    print(f"\n{Colors.BLUE}⚙️ Optimizing Environment...{Colors.ENDC}")
    
    use_gpu = False
    gpus = tf.config.list_physical_devices("GPU")
    
    if force_cpu_if_low and image_count < BATCH_THRESHOLD_FOR_GPU:
        print(f"{Colors.GREEN}⚡ Workload is small ({image_count} imgs) -> Using CPU for speed.{Colors.ENDC}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"{Colors.GREEN}🚀 Workload is large ({image_count} imgs) -> Using GPU ({gpus[0].name}).{Colors.ENDC}")
            use_gpu = True
        except:
            print(f"{Colors.WARNING}⚠️ GPU Init failed -> Fallback to CPU.{Colors.ENDC}")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        print(f"{Colors.WARNING}⚠️ No GPU found -> Using CPU.{Colors.ENDC}")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return "retinaface" if use_gpu else "mtcnn", use_gpu

def main():
    print_header()

    # 1. Validation
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        print(f"{Colors.WARNING}⚠️ '{UPLOAD_DIR}' folder was missing. Created it.{Colors.ENDC}")
        print(f"{Colors.CYAN}👉 Please put new identity folders inside '{UPLOAD_DIR}' and run again.{Colors.ENDC}")
        return

    new_folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
    if not new_folders:
        print(f"{Colors.WARNING}⚠️ No new folders found in '{UPLOAD_DIR}'.{Colors.ENDC}")
        return

    # 2. Model Selection
    model_name, model_dir_name = get_model_choice()
    
    MODEL_DIR = os.path.join("models", model_dir_name)
    FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
    LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(LABELS_PATH):
        print(f"{Colors.FAIL}❌ Model files not found in '{MODEL_DIR}'. Please run 'train_model.py' first.{Colors.ENDC}")
        return

    # 3. Scan Images
    print(f"\n{Colors.BLUE}📂 Scanning New Content...{Colors.ENDC}")
    image_paths = []
    for person in new_folders:
        person_path = os.path.join(UPLOAD_DIR, person)
        for img in os.listdir(person_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append((person, os.path.join(person_path, img)))
    
    total_images = len(image_paths)
    print(f"   👤 New Identities : {len(new_folders)}")
    print(f"   📸 New Images     : {total_images}")

    if total_images == 0:
        print(f"{Colors.FAIL}❌ No valid images found.{Colors.ENDC}")
        return

    # 4. Device Config
    DETECTOR, _ = configure_device(total_images)

    # 5. Load Existing Model
    print(f"\n{Colors.BLUE}🔄 Loading Existing Index...{Colors.ENDC}")
    index = faiss.read_index(FAISS_INDEX_PATH)
    labels = list(np.load(LABELS_PATH))
    print(f"   ✅ Loaded {len(labels)} existing embeddings.")

    # 6. Process New Images
    print(f"\n{Colors.BLUE}🚀 Extracting Embeddings ({model_name})...{Colors.ENDC}")
    
    new_embeddings = []
    new_labels = []
    skipped = 0
    start_time = time.time()

    # Pre-load model
    try:
        DeepFace.represent(img_path=np.zeros((224,224,3), dtype=np.uint8), model_name=model_name, detector_backend=DETECTOR, enforce_detection=False)
    except: pass

    for i, (person, img_path) in enumerate(image_paths):
        # Progress
        sys.stdout.write(f"\r   Processing {i+1}/{total_images} : {person}")
        sys.stdout.flush()

        try:
            with redirect_stdout(StringIO()):
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=DETECTOR,
                    enforce_detection=True
                )
            
            emb = np.array(rep[0]["embedding"], dtype="float32")
            emb /= np.linalg.norm(emb)
            
            new_embeddings.append(emb)
            new_labels.append(person)
        except:
            skipped += 1

    print(f"\n\n{Colors.BLUE}💾 Updating Files...{Colors.ENDC}")

    if new_embeddings:
        # Update FAISS
        new_embeddings = np.vstack(new_embeddings).astype("float32")
        index.add(new_embeddings)
        labels.extend(new_labels)
        
        # Save Model
        faiss.write_index(index, FAISS_INDEX_PATH)
        np.save(LABELS_PATH, np.array(labels))
        print(f"   ✅ Model Updated.")

        # Move files to dataset
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
        
        for person in new_folders:
            src = os.path.join(UPLOAD_DIR, person)
            dst = os.path.join(DATASET_DIR, person)
            
            # Merge logic
            if os.path.exists(dst):
                for f in os.listdir(src):
                    shutil.move(os.path.join(src, f), os.path.join(dst, f))
                os.rmdir(src) # Remove empty folder
            else:
                shutil.move(src, dst)
        
        print(f"   ✅ Images moved to '{DATASET_DIR}'.")
        print(f"   🗑️ Cleaned up '{UPLOAD_DIR}'.")
        
    else:
        print(f"{Colors.FAIL}❌ No valid faces found in new images.{Colors.ENDC}")

    total_time = time.time() - start_time
    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ SUCCESS!{Colors.ENDC}")
    print(f"   ➕ Added Vectors : {len(new_embeddings)}")
    print(f"   ⏱️ Time Taken    : {total_time:.2f} sec")

if __name__ == "__main__":
    main()
