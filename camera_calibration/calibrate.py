import cv2
import mediapipe as mp
import numpy as np
import time
import json

import os

# ================= CONFIG =================
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")
DOT_RADIUS = 14
MOVE_ALPHA = 0.18
HOLD_TIME = 2.5
MARGIN_RATIO = 0.08

# ================= MEDIAPIPE =================
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

# ================= HELPERS =================
def iris_center(lm, idx, shape):
    h, w = shape[:2]
    return np.mean([[lm[i].x * w, lm[i].y * h] for i in idx], axis=0)

def eye_ratio(iris, corners):
    w = np.linalg.norm(corners[1] - corners[0])
    c = corners.mean(axis=0)
    return (iris - c) / (w / 2 + 1e-6)

def get_ratio_and_draw(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    li = iris_center(lm, LEFT_IRIS, frame.shape)
    ri = iris_center(lm, RIGHT_IRIS, frame.shape)
    lc = iris_center(lm, LEFT_EYE_CORNERS, frame.shape)
    rc = iris_center(lm, RIGHT_EYE_CORNERS, frame.shape)

    cv2.circle(frame, li.astype(int), 4, (0,255,0), -1)
    cv2.circle(frame, ri.astype(int), 4, (0,255,0), -1)

    return (eye_ratio(li, lc) + eye_ratio(ri, rc)) / 2

def zone_position(zone, shape):
    h, w = shape[:2]
    mx = int(w * MARGIN_RATIO)
    my = int(h * MARGIN_RATIO)
    return {
        "CENTER": (w//2, h//2),
        "LEFT":   (mx, h//2),
        "RIGHT":  (w-mx, h//2),
        "UP":     (w//2, my),
        "DOWN":   (w//2, h-my),
    }[zone]

def smooth(curr, target):
    return (
        int(curr[0] + MOVE_ALPHA * (target[0] - curr[0])),
        int(curr[1] + MOVE_ALPHA * (target[1] - curr[1]))
    )

# ================= MAIN =================
def run():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "Calibration",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    steps = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    data = {k: [] for k in steps}
    dot_pos = None

    print("\n🎯 Eye Gaze Calibration")
    print("Follow the RED DOT with your eyes only")
    print("Green dots MUST stay on eyes\n")

    for step in steps:
        print(f"➡️ Look {step}")
        start = time.time()

        while time.time() - start < HOLD_TIME:
            ok, frame = cap.read()
            if not ok:
                continue

            if dot_pos is None:
                dot_pos = zone_position("CENTER", frame.shape)

            dot_pos = smooth(dot_pos, zone_position(step, frame.shape))

            r = get_ratio_and_draw(frame)
            if r is not None:
                data[step].append(r)

            cv2.circle(frame, dot_pos, DOT_RADIUS, (0,0,255), -1)
            cv2.imshow("Calibration", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    # ================= COMPUTE CALIB =================
    center = np.mean(data["CENTER"], axis=0)

    calib = {
        "center": center.tolist(),
        "left":  float(np.percentile(np.array(data["LEFT"])[:,0] - center[0], 10)),
        "right": float(np.percentile(np.array(data["RIGHT"])[:,0] - center[0], 90)),
        "up":    float(np.percentile(np.array(data["UP"])[:,1] - center[1], 10)),
        "down":  float(np.percentile(np.array(data["DOWN"])[:,1] - center[1], 90)),
    }


    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib, f, indent=2)

    # ================= PRINT DEBUG =================
    print("\n📊 Calibration Results (IMPORTANT)")
    print(f"CENTER      : {center}")
    print(f"LEFT delta  : {calib['left']:.4f}")
    print(f"RIGHT delta : {calib['right']:.4f}")
    print(f"UP delta    : {calib['up']:.4f}")
    print(f"DOWN delta  : {calib['down']:.4f}")

    print("\n✅ Calibration saved to calibration.json")

if __name__ == "__main__":
    run()
