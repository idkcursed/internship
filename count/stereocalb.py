import cv2
import numpy as np

# ================= USER SETTINGS =================
CAM1_VIDEO = "calib1.mp4"
CAM2_VIDEO = "calib2.mp4"

CHECKERBOARD = (6, 4)   # inner corners
SQUARE_SIZE = 22        # mm
FRAME_STEP = 15         # take every Nth frame
# =================================================

# ---------- Prepare object points ----------
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints1 = []
imgpoints2 = []

cap1 = cv2.VideoCapture(CAM1_VIDEO)
cap2 = cv2.VideoCapture(CAM2_VIDEO)

frame_id = 0
img_size = None

print("[INFO] Reading videos and detecting corners...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    if frame_id % FRAME_STEP == 0:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        img_size = gray1.shape[::-1]

        ok1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD)
        ok2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD)

        # KEEP ONLY MATCHING PAIRS
        if ok1 and ok2:
            objpoints.append(objp)
            imgpoints1.append(corners1)
            imgpoints2.append(corners2)

    frame_id += 1

cap1.release()
cap2.release()

print("Valid stereo pairs:", len(objpoints))

if len(objpoints) < 10:
    raise RuntimeError("Not enough valid stereo pairs")

# ---------- Single-camera calibration ----------
print("[INFO] Calibrating camera 1...")
_, K1, D1, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints1, img_size, None, None
)

print("[INFO] Calibrating camera 2...")
_, K2, D2, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints2, img_size, None, None
)

# ---------- Stereo calibration ----------
print("[INFO] Stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC

_, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints1,
    imgpoints2,
    K1, D1,
    K2, D2,
    img_size,
    flags=flags
)

# ---------- Results ----------
print("\n=== STEREO RESULTS ===")
print("Baseline (mm):", np.linalg.norm(T))
print("Translation T:\n", T)
print("Rotation R:\n", R)

# ---------- Save ----------
np.savez(
    "stereo_calibration.npz",
    K1=K1, D1=D1,
    K2=K2, D2=D2,
    R=R, T=T
)

print("\nStereo calibration complete and saved.")
