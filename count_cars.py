import cv2
import time
import numpy as np
import torch
from rfdetr import RFDETRMedium

# -----------------------------
# CONFIG
# -----------------------------
RTSP_URL = "rtsp://admin:clancy252629@192.168.105.120:554/cam/realmonitor?channel=1&subtype=2"

VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus"}

INFER_INTERVAL = 0.5  # 2 FPS

# -----------------------------
# SIMPLE TRACKER STATE
# -----------------------------
next_id = 0
tracks = {}  # id -> centroid
MAX_DIST = 80  # pixel distance threshold

def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def match_or_create_tracks(centroids):
    global next_id, tracks

    assigned_ids = []
    used_track_ids = set()

    for c in centroids:
        best_id = None
        best_dist = float("inf")

        for tid, tc in tracks.items():
            if tid in used_track_ids:
                continue

            dist = np.linalg.norm(np.array(c) - np.array(tc))
            if dist < best_dist and dist < MAX_DIST:
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = next_id
            next_id += 1

        tracks[best_id] = c
        used_track_ids.add(best_id)
        assigned_ids.append(best_id)

    return assigned_ids

# -----------------------------
# MODEL SETUP
# -----------------------------
print("Loading model...")
model = RFDETRMedium()
model.optimize_for_inference(dtype=torch.float16)
print("Model loaded.")

# -----------------------------
# RTSP
# -----------------------------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_infer_time = 0
frame_num = 0

# -----------------------------
# PERSISTENT COUNTER
# -----------------------------
seen_vehicle_ids = set()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ok, frame = cap.read()

    if not ok:
        print("Frame dropped — retrying")
        time.sleep(0.01)
        continue

    now = time.time()

    if now - last_infer_time < INFER_INTERVAL:
        time.sleep(0.01)
        continue

    last_infer_time = now

    # -------------------------
    # INFERENCE
    # -------------------------
    detections = model.predict(frame, threshold=0.5)

    class_names = detections.data["class_name"]
    xyxy = detections.xyxy

    vehicle_mask = np.isin(class_names, list(VEHICLE_CLASSES))

    vehicle_boxes = xyxy[vehicle_mask]

    centroids = [get_centroid(b) for b in vehicle_boxes]

    # -------------------------
    # TRACKING
    # -------------------------
    ids = match_or_create_tracks(centroids)

    for tid in ids:
        seen_vehicle_ids.add(tid)

    VEHICLES = len(ids)
    TOTAL_VEHICLES = len(seen_vehicle_ids)

    # -------------------------
    # LOGGING
    # -------------------------
    print(
        f"[{frame_num}] "
        f"VEHICLES: {VEHICLES} | "
        f"TOTAL_VEHICLES: {TOTAL_VEHICLES}"
    )

    # -------------------------
    # DEBUG SNAPSHOT
    # -------------------------
    if frame_num % 20 == 0:
        cv2.imwrite(f"debug/frame_{frame_num}.jpg", frame)

    frame_num += 1
