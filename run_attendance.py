#!/usr/bin/env python3
"""
ANTI-SPOOF FACE ATTENDANCE SYSTEM
Smooth Camera + ArcFace + FAISS + Eye Tracking
FAST + FULL TIMING + IDENTITY LOCK
"""

# ================== SILENCE LOGS ==================
import os, logging, warnings, sys
sys.stdout.reconfigure(encoding='utf-8')
from contextlib import redirect_stdout
from io import StringIO

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

print("🔕 Logs hidden | Smooth Anti-Spoof Attendance (FAST)")
# ================== COLORS ==================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    MAGENTA = '\033[35m'

# ================== IMPORTS ==================
import cv2
import time
import json
import random
import numpy as np
import faiss
import threading
import sys

from deepface import DeepFace
import mediapipe as mp

# ================== CONFIG ==================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MODEL_DIR = os.path.join("models", "arcface")
FAISS_INDEX = "faiss_index.bin"
LABELS_FILE = "labels.npy"
CALIB_FILE = os.path.join("camera_calibration", "calibration.json")

MODEL_NAME = "ArcFace"
DETECTOR = "mtcnn"

RECOG_THRESHOLD = 1.25
FRAME_SKIP = 5

REQUIRED_MATCHES = 2      # 🔥 FAST LOCK (was 12)

# Eye liveness
NUM_DIRECTIONS = 2
HOLD_TIME = 2.0
PASS_THRESHOLD = 0.6
MOVE_ALPHA = 0.2
DOT_RADIUS = 14
MARGIN_RATIO = 0.08

# ================== GLOBAL TIMERS ==================
SCRIPT_START = time.time()
RECOG_START = None
LIVENESS_START = None

# ================== LOAD MODELS ==================
print("🔄 Loading models...")

index = faiss.read_index(os.path.join(MODEL_DIR, FAISS_INDEX))
labels = np.load(os.path.join(MODEL_DIR, LABELS_FILE))

with open(CALIB_FILE) as f:
    CALIB = json.load(f)

CENTER = np.array(CALIB["center"])
CALIB_RANGE = {
    "LEFT": abs(CALIB["left"]),
    "RIGHT": abs(CALIB["right"]),
    "UP": abs(CALIB["up"]),
    "DOWN": abs(CALIB["down"]),
}

print("✅ Models loaded")

# Stats
total_embeddings = index.ntotal
total_identities = len(np.unique(labels))
print(f"📊 Stats: {total_identities} Identities | {total_embeddings} Total Images")


# ================== MEDIAPIPE ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# ================== EYE HELPERS ==================
def iris_center(lm, idx, shape):
    h, w = shape[:2]
    return np.mean([[lm[i].x*w, lm[i].y*h] for i in idx], axis=0)

def eye_ratio(iris, corners):
    w = np.linalg.norm(corners[1] - corners[0])
    c = corners.mean(axis=0)
    return (iris - c) / (w/2 + 1e-6)

def get_eye_ratio(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    li = iris_center(lm, LEFT_IRIS, frame.shape)
    ri = iris_center(lm, RIGHT_IRIS, frame.shape)
    lc = iris_center(lm, LEFT_EYE_CORNERS, frame.shape)
    rc = iris_center(lm, RIGHT_EYE_CORNERS, frame.shape)

    return (eye_ratio(li, lc) + eye_ratio(ri, rc)) / 2

def zone_position(zone, shape):
    h, w = shape[:2]
    mx, my = int(w*MARGIN_RATIO), int(h*MARGIN_RATIO)
    return {
        "LEFT": (mx, h//2),
        "RIGHT": (w-mx, h//2),
        "UP": (w//2, my),
        "DOWN": (w//2, h-my)
    }[zone]

def smooth(curr, target):
    return (
        int(curr[0] + MOVE_ALPHA*(target[0]-curr[0])),
        int(curr[1] + MOVE_ALPHA*(target[1]-curr[1]))
    )

# ================== EYE LIVENESS ==================
def run_eye_liveness(cap):
    global LIVENESS_START
    LIVENESS_START = time.time()

    dirs = random.sample(["LEFT","RIGHT","UP","DOWN"], NUM_DIRECTIONS)
    print("\n👁️ Eye liveness started")
    print("➡️ Directions :", dirs)

    scores, dot_pos = [], None

    for expected in dirs:
        start, values = time.time(), []

        while time.time() - start < HOLD_TIME:
            ok, frame = cap.read()
            if not ok:
                continue

            if dot_pos is None:
                dot_pos = (frame.shape[1]//2, frame.shape[0]//2)

            dot_pos = smooth(dot_pos, zone_position(expected, frame.shape))
            r = get_eye_ratio(frame)

            if r is not None:
                dx, dy = (r - CENTER)
                dx = -dx
                values.append(dx if expected in ["LEFT","RIGHT"] else dy)

            cv2.circle(frame, dot_pos, DOT_RADIUS, (0,0,255), -1)
            cv2.imshow("Attendance", frame)
            cv2.waitKey(1)

        vals = np.array(values) if values else np.array([0])
        calib = CALIB_RANGE[expected] + 1e-6
        movement = abs(np.min(vals)) if expected in ["LEFT","UP"] else abs(np.max(vals))
        scores.append(min(movement / calib, 1.0))

    final = float(np.mean(scores))
    print(f"📊 Liveness score : {final*100:.1f}%")
    print(f"⏱️ Eye liveness time : {time.time()-LIVENESS_START:.2f} sec")
    return final >= PASS_THRESHOLD

# ================== RECOGNITION THREAD ==================
last_name = "Searching..."
match_count = 0
processing = False
identity_locked = False

def recognize(frame_small):
    global last_name, match_count, processing, identity_locked, RECOG_START

    if identity_locked:
        processing = False
        return

    if RECOG_START is None:
        RECOG_START = time.time()

    try:
        t0 = time.time()
        with redirect_stdout(StringIO()):
            rep = DeepFace.represent(
                frame_small,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=False
            )
        t1 = time.time()

        if not rep:
            match_count = 0
            return

        emb = np.array(rep[0]["embedding"], dtype="float32")
        emb /= np.linalg.norm(emb)

        t2 = time.time()
        D, I = index.search(emb.reshape(1,-1), 1)
        t3 = time.time()

        dist = float(D[0][0])

        print("\n🔍 Face recognition cycle")
        print(f"🧠 Embedding time : {t1-t0:.2f} sec")
        print(f"⚡ FAISS search   : {t3-t2:.4f} sec")
        print(f"⏱️ Cycle total   : {t3-t0:.2f} sec")

        if dist < RECOG_THRESHOLD:
            name = labels[I[0][0]]
            if name == last_name:
                match_count += 1
            else:
                last_name = name
                match_count = 1
        else:
            match_count = 0
            last_name = "Searching..."

        print(f"⏳ Matches : {match_count}/{REQUIRED_MATCHES}")

        if match_count >= REQUIRED_MATCHES:
            identity_locked = True

    finally:
        processing = False

# ================== MAIN ==================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Attendance", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Attendance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\n🎥 Attendance system running...\n")

frame_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame_id += 1

    if frame_id % FRAME_SKIP == 0 and not processing and not identity_locked:
        processing = True
        small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        threading.Thread(target=recognize, args=(small,), daemon=True).start()

    cv2.putText(frame, last_name, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0) if last_name!="Searching..." else (0,0,255), 2)

    cv2.imshow("Attendance", frame)

    if identity_locked:
        print(f"\n{Colors.GREEN}{Colors.BOLD}👤 Identity Locked: {last_name}{Colors.ENDC}")
        passed = run_eye_liveness(cap)

        TOTAL_TIME = time.time() - SCRIPT_START
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}==================== SUMMARY ===================={Colors.ENDC}")
        print(f"{Colors.CYAN}👤 Identity          : {Colors.BOLD}{last_name}{Colors.ENDC}")
        print(f"{Colors.BLUE}🧠 Recognition Time  : {Colors.BOLD}{time.time()-RECOG_START:.2f} sec{Colors.ENDC}")
        print(f"{Colors.MAGENTA}👁️  Liveness Time     : {Colors.BOLD}{time.time()-LIVENESS_START:.2f} sec{Colors.ENDC}")
        print(f"{Colors.GREEN}⏱️  Total Runtime     : {Colors.BOLD}{TOTAL_TIME:.2f} sec{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}================================================{Colors.ENDC}")

        if passed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ ATTENDANCE MARKED: {last_name}{Colors.ENDC}")
        else:
            print(f"\n{Colors.FAIL}{Colors.BOLD}❌ SPOOF DETECTED: IDENTITY REJECTED{Colors.ENDC}")

        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
 