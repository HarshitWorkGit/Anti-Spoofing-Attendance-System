#!/usr/bin/env python3
"""
Interactive Face Recognition Training Script
Supports ArcFace & FaceNet
CPU & GPU compatible with Fail-Safe
"""

import os
import sys
import time
import numpy as np
import faiss
import tensorflow as tf
from deepface import DeepFace
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO

sys.stdout.reconfigure(encoding='utf-8')

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

def print_header():
    print(f"\n{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}   🤖 FACE RECOGNITION TRAINING CLI   {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}========================================{Colors.ENDC}\n")

# ================= INPUTS =================
def get_user_choice():
    print_header()
    
    # --- MODEL SELECTION ---
    print(f"{Colors.CYAN}Select Model:{Colors.ENDC}")
    print("  [a] ArcFace (Recommended)")
    print("  [f] FaceNet")
    while True:
        m = input(f"{Colors.BOLD}>> Enter choice (a/f): {Colors.ENDC}").strip().lower()
        if m in ['a', 'f']:
            break
        print(f"{Colors.FAIL}Invalid input! Please enter 'a' or 'f'.{Colors.ENDC}")
    
    model_name = "ArcFace" if m == 'a' else "FaceNet"
    model_folder = "arcface" if m == 'a' else "facenet"

    # --- DEVICE SELECTION ---
    print(f"\n{Colors.CYAN}Select Processing Device:{Colors.ENDC}")
    print("  [c] CPU (Stable)")
    print("  [g] GPU (Fast - Requires CUDA)")
    while True:
        d = input(f"{Colors.BOLD}>> Enter choice (c/g): {Colors.ENDC}").strip().lower()
        if d in ['c', 'g']:
            break
        print(f"{Colors.FAIL}Invalid input! Please enter 'c' or 'g'.{Colors.ENDC}")
    
    use_gpu = (d == 'g')

    return model_name, model_folder, use_gpu

# ================= CONFIG =================
def configure_environment(use_gpu):
    print(f"\n{Colors.BLUE}⚙️ Configuring Environment...{Colors.ENDC}")
    
    # Hardware Check
    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0

    if use_gpu and not gpu_available:
        print(f"{Colors.WARNING}⚠️ GPU requested but NOT FOUND!{Colors.ENDC}")
        print(f"{Colors.WARNING}   ➔ Switching to CPU automatically.{Colors.ENDC}")
        use_gpu = False
    
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(f"{Colors.GREEN}✅ Using Device: CPU{Colors.ENDC}")
    else:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"{Colors.GREEN}✅ Using Device: GPU ({gpus[0].name}){Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ GPU Init Failed: {e}{Colors.ENDC}")
            print(f"{Colors.WARNING}   ➔ Fallback to CPU.{Colors.ENDC}")
            use_gpu = False
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return use_gpu

# ================= TRAINING LOGIC =================
def train():
    try:
        model_name, model_folder, use_gpu = get_user_choice()
        use_gpu = configure_environment(use_gpu)

        DATASET_PATH = "dataset"
        MODEL_BASE_DIR = "models"
        TARGET_DIR = os.path.join(MODEL_BASE_DIR, model_folder)
        
        # Ensure target directory exists
        os.makedirs(TARGET_DIR, exist_ok=True)

        # Detector Config
        DETECTOR = "retinaface" if use_gpu else "mtcnn"
        BATCH_SIZE = 48 if use_gpu else 24
        IMAGE_EXTS = (".jpg", ".jpeg", ".png")

        # ================= LOAD DATA =================
        print(f"\n{Colors.BLUE}📂 Scanning Dataset...{Colors.ENDC}")
        
        if not os.path.exists(DATASET_PATH):
            print(f"{Colors.FAIL}❌ Error: 'dataset' folder not found!{Colors.ENDC}")
            return # Exit cleanly without touching models

        image_paths = []
        for person in sorted(os.listdir(DATASET_PATH)):
            person_dir = os.path.join(DATASET_PATH, person)
            if not os.path.isdir(person_dir):
                continue
            for img in os.listdir(person_dir):
                if img.lower().endswith(IMAGE_EXTS):
                    image_paths.append((person, os.path.join(person_dir, img)))

        total_images = len(image_paths)
        if total_images == 0:
            print(f"{Colors.WARNING}⚠️ Dataset is empty! No images found in '{DATASET_PATH}'.{Colors.ENDC}")
            print(f"{Colors.WARNING}   ➔ No changes were made to existing models.{Colors.ENDC}")
            return # Exit cleanly

        identities = len(set(p for p, _ in image_paths))

        print(f"   👤 Identities Found : {identities}")
        print(f"   📸 Total Images     : {total_images}")
        
        confirm = input(f"\n{Colors.BOLD}>> Start Training? (y/n): {Colors.ENDC}").strip().lower()
        if confirm != 'y':
            print(f"{Colors.WARNING}🚫 Training cancelled.{Colors.ENDC}")
            return

        # ================= EMBEDDING =================
        print(f"\n{Colors.BLUE}🚀 Starting Embedding Extraction ({model_name})...{Colors.ENDC}")
        
        embeddings = []
        labels = []
        skipped = 0
        start_time = time.time()
        
        # Initialize DeepFace once to download weights if needed
        # We do a dummy call to ensure model is loaded before loop
        try:
           DeepFace.represent(img_path=np.zeros((224,224,3), dtype=np.uint8), model_name=model_name, detector_backend=DETECTOR, enforce_detection=False)
        except:
           pass

        num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(range(num_batches), desc="Processing", unit="batch", ncols=100)

        for batch_idx in pbar:
            batch = image_paths[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

            for person, img_path in batch:
                try:
                    # Silence DeepFace logs per image
                    with redirect_stdout(StringIO()):
                        rep = DeepFace.represent(
                            img_path=img_path,
                            model_name=model_name,
                            detector_backend=DETECTOR,
                            enforce_detection=True
                        )

                    emb = np.array(rep[0]["embedding"], dtype="float32")
                    emb /= np.linalg.norm(emb) # Normalize

                    embeddings.append(emb)
                    labels.append(person)

                except Exception:
                    skipped += 1
            
            # Stats update
            elapsed = time.time() - start_time
            speed = len(embeddings) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({"img/s": f"{speed:.1f}", "skip": skipped})

        pbar.close()

        # ================= SAVE =================
        if len(embeddings) == 0:
            print(f"\n{Colors.FAIL}❌ No embeddings generated! Check your images.{Colors.ENDC}")
            return

        print(f"\n{Colors.BLUE}💾 Saving Model (FAISS Index)...{Colors.ENDC}")
        
        embeddings = np.vstack(embeddings).astype("float32")
        
        # Create Index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # Save paths
        index_path = os.path.join(TARGET_DIR, "faiss_index.bin")
        labels_path = os.path.join(TARGET_DIR, "labels.npy")

        faiss.write_index(index, index_path)
        np.save(labels_path, np.array(labels))

        total_time = time.time() - start_time
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ TRAINING SUCCESSFUL!{Colors.ENDC}")
        print(f"   ⏱️ Total Time : {total_time/60:.2f} min")
        print(f"   📂 Model Saved: {TARGET_DIR}")
        print(f"   🔹 Identities : {identities}")
        print(f"   🔹 Embeddings : {len(embeddings)}")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.FAIL}🛑 Process Interrupted by User.{Colors.ENDC}")
        print(f"{Colors.WARNING}   ⚠️ No changes were saved to existing models.{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ An Unexpected Error Occurred:{Colors.ENDC}")
        print(e)

if __name__ == "__main__":
    train()
