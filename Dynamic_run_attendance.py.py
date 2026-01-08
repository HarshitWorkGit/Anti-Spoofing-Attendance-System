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

# ================== MODEL SELECTION ==================
print("\nSelect face recognition model: ")
print("Press 'a' for ArcFace")
print("Press 'f' for FaceNet512")

choice = input("Your choice (a/f): ").strip().lower()

if choice == "f":
    MODEL_NAME = "Facenet512"
    MODEL_FOLDER = "facenet"
    print("✅ FaceNet512 selected")
else:
    MODEL_NAME = "ArcFace"
    MODEL_FOLDER = "arcface"
    print("✅ ArcFace selected (default)")


# ================== CONFIG ==================
CAMERA_INDEX = 0
FRAME_WIDTH = 320    # 🚀 OPTIMIZATION: process smaller frames
FRAME_HEIGHT = 240

MODEL_DIR = os.path.join("models", MODEL_FOLDER)

# MODEL_NAME = "Facenet512"

FAISS_INDEX = "faiss_index.bin"
LABELS_FILE = "labels.npy"
CALIB_FILE = os.path.join("camera_calibration", "calibration.json")

# DETECTOR = "mtcnn 
DETECTOR = "opencv"  # 🚀 OPTIMIZATION: 'opencv' is 10x faster than 'mtcnn' on CPU

RECOG_THRESHOLD = 1
FRAME_SKIP = 5

REQUIRED_MATCHES = 2      # 🔥 FAST LOCK

# Eye liveness
NUM_DIRECTIONS = 2       # 2 for fast, 3 for accurate.
HOLD_TIME = 1.5          # 🚀 OPTIMIZATION: Reduced from 2.0s (change accordingly)
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

# Eye landmarks for blink detection (Eye Aspect Ratio)
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]

# ================== EYE HELPERS ==================
def eye_aspect_ratio(eye_points, landmarks, shape):
    """Calculate Eye Aspect Ratio (EAR) for blink detection"""
    h, w = shape[:2]
    coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in eye_points])
    
    # Vertical distances
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    
    # Horizontal distance
    h_dist = np.linalg.norm(coords[0] - coords[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h_dist + 1e-6)
    return ear

def detect_blink(frame):
    """Returns True if a blink is detected (EAR < threshold)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return False
    
    lm = res.multi_face_landmarks[0].landmark
    left_ear = eye_aspect_ratio(LEFT_EYE_POINTS, lm, frame.shape)
    right_ear = eye_aspect_ratio(RIGHT_EYE_POINTS, lm, frame.shape)
    
    avg_ear = (left_ear + right_ear) / 2.0
    # EAR < 0.2 typically indicates closed eyes
    return avg_ear < 0.2
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

    pass_count = 0
    dot_pos = None
    blink_detected = False
    prev_blink_state = False

    for expected in dirs:
        start_t = time.time()
        eye_vals, dot_vals = [], []
        
        while time.time() - start_t < HOLD_TIME:
            ok, frame = cap.read()
            if not ok: continue
            
            h, w = frame.shape[:2]
            cx, cy = w//2, h//2
            
            if dot_pos is None:
                dot_pos = (cx, cy)
                
            # Move dot
            target = zone_position(expected, frame.shape)
            dot_pos = smooth(dot_pos, target)
            
            # Blink detection (bonus anti-spoofing)
            is_blinking = detect_blink(frame)
            if is_blinking and not prev_blink_state:
                blink_detected = True
            prev_blink_state = is_blinking
            
            # Eye tracking
            r = get_eye_ratio(frame)
            if r is not None:
                dx, dy = (r - CENTER)
                
                # Get relevant axis values
                e_v = dx if expected in ["LEFT","RIGHT"] else dy
                d_v = (dot_pos[0] - cx) if expected in ["LEFT","RIGHT"] else (dot_pos[1] - cy)
                
                eye_vals.append(e_v)
                dot_vals.append(d_v)
                
            cv2.circle(frame, dot_pos, DOT_RADIUS, (0,0,255), -1)
            cv2.imshow("Attendance", frame)
            cv2.waitKey(1)

        # ========== ANALYSIS ==========
        if len(eye_vals) < 15:
            print(f"   ⚠️ [{expected}] Not enough data (only {len(eye_vals)} frames)")
            continue

        e_arr = np.array(eye_vals)
        d_arr = np.array(dot_vals)
        
        # === CHECK 1: Variance (Static Image Detection) ===
        eye_variance = np.std(e_arr)
        variance_pass = eye_variance > 0.005  # Static images have ~0 variance
        # Increase to 0.008 if too strict ----------------<>


        # === CHECK 2: Correlation (Following the Dot) ===
        if np.std(e_arr) < 1e-4 or np.std(d_arr) < 1e-4:
            corr = 0
        else:
            corr = np.corrcoef(e_arr, d_arr)[0,1]
        
        # Relaxed threshold: 0.3 allows for noisy webcam tracking
        correlation_pass = corr > 0.3           # Decrease to 0.25 if too strict ----------------<>
        
        # === CHECK 3: Amplitude (Movement in Correct Direction) ===
        if expected in ["LEFT", "UP"]:
            # Expect negative deviation
            dev = abs(min(0, np.min(e_arr)))
        else:
            # Expect positive deviation
            dev = max(0, np.max(e_arr))
            
        calib = CALIB_RANGE[expected] + 1e-6
        amp_score = min(dev / calib, 1.0)
        
        # Relaxed threshold: 0.15 allows for smaller movements
        amplitude_pass = amp_score > 0.15        # Decrease to 0.10 if too strict ----------------<>
        
        # === COMPOSITE SCORING: 2 out of 3 checks must pass ===
        checks_passed = sum([variance_pass, correlation_pass, amplitude_pass])
        direction_pass = checks_passed >= 2
        
        if direction_pass:
            pass_count += 1
        
        # Debug output
        print(f"   [{expected}] Var:{eye_variance:.4f}{'✓' if variance_pass else '✗'} | "
              f"Corr:{corr:.2f}{'✓' if correlation_pass else '✗'} | "
              f"Amp:{amp_score:.2f}{'✓' if amplitude_pass else '✗'} | "
              f"Result:{'PASS' if direction_pass else 'FAIL'} ({checks_passed}/3)")

    # Final result
    success = (pass_count == NUM_DIRECTIONS)
    
    # Blink bonus (not required, but adds confidence)
    blink_bonus = " + Blink✓" if blink_detected else " (No blink detected)"
    
    print(f"\n📊 Liveness Result : {'PASS' if success else 'FAIL'} ({pass_count}/{NUM_DIRECTIONS}){blink_bonus}")
    print(f"⏱️ Eye liveness time : {time.time()-LIVENESS_START:.2f} sec")
    
    # Note: We don't require blink, but it's a good sign
    return success


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


# -------- Force window to front (Windows fix) --------
try:
    import win32gui, win32con
    hwnd = win32gui.FindWindow(None, "Attendance")
    if hwnd != 0:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.SetForegroundWindow(hwnd)
except:
    pass


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
 