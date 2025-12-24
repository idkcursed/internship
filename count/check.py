from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
cap = cv2.VideoCapture("detect1.mp4")

ret, frame = cap.read()
cap.release()

results = model(frame)[0]
print("Detections:", len(results.boxes))
print(results.boxes.xyxy)
print(results.boxes.conf)


