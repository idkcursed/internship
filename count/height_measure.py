import cv2
import numpy as np
from ultralytics import YOLO
from statistics import median

# ================= SETTINGS =================
CAM1_VIDEO = "detect1.mp4"
CAM2_VIDEO = "detect2.mp4"
CALIB_FILE = "stereo_calibration.npz"
MODEL_PATH = "best.pt"

CONF_THRESH = 0.3
FRAME_STEP = 3
MAX_FRAMES = 40

# --- BBOX CORRECTION (CRITICAL) ---
TOP_CROP = 0.11     # remove YOLO top padding (8%)
BOTTOM_CROP = 0.26  # remove YOLO bottom padding (12%)
# ==========================================

# ---------- Load calibration ----------
data = np.load(CALIB_FILE)
K1, D1 = data["K1"], data["D1"]
K2, D2 = data["K2"], data["D2"]
R, T = data["R"], data["T"]

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))

# ---------- Load YOLO ----------
model = YOLO(MODEL_PATH)

# ---------- Videos ----------
cap1 = cv2.VideoCapture(CAM1_VIDEO)
cap2 = cv2.VideoCapture(CAM2_VIDEO)

frame_id = 0
heights = []

print("[INFO] Measuring height (bias-corrected)...")

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    if not ret1 or not ret2:
        break

    if frame_id % FRAME_STEP != 0:
        frame_id += 1
        continue

    f1 = cv2.undistort(f1, K1, D1)
    f2 = cv2.undistort(f2, K2, D2)

    r1 = model(f1, conf=CONF_THRESH, verbose=False)[0]
    r2 = model(f2, conf=CONF_THRESH, verbose=False)[0]

    if len(r1.boxes) == 0 or len(r2.boxes) == 0:
        frame_id += 1
        continue

    b1 = r1.boxes.xyxy.cpu().numpy()[0]
    b2 = r2.boxes.xyxy.cpu().numpy()[0]

    x1,y1,x2,y2 = b1
    x1p,y1p,x2p,y2p = b2

    h1 = y2 - y1
    h2 = y2p - y1p

    # -------- BBOX SHRINK (KEY FIX) --------
    top1 = ((x1+x2)/2, y1 + TOP_CROP * h1)
    bot1 = ((x1+x2)/2, y2 - BOTTOM_CROP * h1)

    top2 = ((x1p+x2p)/2, y1p + TOP_CROP * h2)
    bot2 = ((x1p+x2p)/2, y2p - BOTTOM_CROP * h2)
    # --------------------------------------

    pts1 = np.array([top1, bot1], dtype=np.float32).T
    pts2 = np.array([top2, bot2], dtype=np.float32).T

    pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts3d = pts4d[:3] / pts4d[3]

    height_mm = abs(pts3d[1,0] - pts3d[1,1])
    heights.append(height_mm)

    print(f"Frame {frame_id}: {height_mm:.2f} mm")

    if len(heights) >= MAX_FRAMES:
        break

    frame_id += 1

cap1.release()
cap2.release()

# ---------- Final result ----------
if heights:
    print("\n==============================")
    print(f"FINAL HEIGHT = {median(heights):.2f} mm")
    print("==============================")
else:
    print("❌ No valid height measurements")


