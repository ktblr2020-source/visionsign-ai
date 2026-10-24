import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR = './dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TARGET_LETTER = 'Y'  # Change this letter as you collect each one
dataset_path = os.path.join(DATA_DIR, TARGET_LETTER)
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cap = cv2.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

print(f"\n--- COLLECTING NORMALIZED DATA FOR: {TARGET_LETTER} ---")
print("Press 's' to save snapshots. Move your hand slightly around while saving!")
print("Press 'q' to finish this letter.")

count = len(os.listdir(dataset_path))
start_time = time.time()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        has_hand = False
        if detection_result.hand_landmarks:
            has_hand = True
            hand_landmarks = detection_result.hand_landmarks[0]
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (168, 85, 247), -1)

        cv2.putText(frame, f"Letter: {TARGET_LETTER} | Saved: {count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Normalized Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and has_hand:
            # RELATIVE COORDINATE NORMALIZATION LOGIC
            landmarks = []
            base_x = hand_landmarks[0].x
            base_y = hand_landmarks[0].y
            base_z = hand_landmarks[0].z
            
            # Make wrist (landmark 0) the origin (0,0,0) and scale relative to max distance
            temp_coords = []
            for lm in hand_landmarks:
                temp_coords.append([lm.x - base_x, lm.y - base_y, lm.z - base_z])
            
            max_dist = max([np.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2) for pt in temp_coords])
            if max_dist == 0: 
                max_dist = 1.0

            for pt in temp_coords:
                landmarks.extend([pt[0]/max_dist, pt[1]/max_dist, pt[2]/max_dist])
            
            file_path = os.path.join(dataset_path, f"{count}.npy")
            np.save(file_path, np.array(landmarks))
            count += 1
            print(f"Saved normalized snapshot {count} for {TARGET_LETTER}")

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()