import cv2
import os
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm

# Parameters
input_path = "input.webm"
output_dir = "extracted_faces"
target_faces = 4000
min_face_size = 80  # to filter too small detections
frame_skip = 10  # initially skip 10 frames at a time

os.makedirs(output_dir, exist_ok=True)
detector = MTCNN()

# Load video
cap = cv2.VideoCapture(input_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps

print(f"Total frames: {total_frames}, Duration: {duration:.2f}s, FPS: {fps}")

saved_faces = 0
frame_count = 0
pbar = tqdm(total=target_faces)

while cap.isOpened() and saved_faces < target_faces:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb_frame)

        for result in results:
            x, y, w, h = result['box']
            if w < min_face_size or h < min_face_size:
                continue  # skip small faces

            face = rgb_frame[y:y + h, x:x + w]
            if face.size == 0:
                continue

            face = cv2.resize(face, (128, 128))
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{output_dir}/face_{saved_faces:04d}.jpg", face_bgr)
            saved_faces += 1
            pbar.update(1)
            if saved_faces >= target_faces:
                break

    frame_count += 1

cap.release()
pbar.close()
print(f"✅ Extracted {saved_faces} faces to `{output_dir}/`")
