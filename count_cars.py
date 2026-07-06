import os
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
DEBUG_DIR = "debug"

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

def draw_vehicle_detections(frame, boxes, ids):
    annotated = frame.copy()

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = get_centroid(box)
        cx, cy = int(cx), int(cy)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        label = f"ID {track_id}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1 - 8, label_h + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_h - 4),
            (x1 + label_w + 4, label_y + baseline),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )

    return annotated

# -----------------------------
# MODEL SETUP
# -----------------------------
print("Loading model...")
model = RFDETRMedium()
model.optimize_for_inference(dtype=torch.float16)
print("Model loaded.")

os.makedirs(DEBUG_DIR, exist_ok=True)

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

    new_ids = [tid for tid in ids if tid not in seen_vehicle_ids]
    for tid in ids:
        seen_vehicle_ids.add(tid)

    current_vehicles = len(ids)
    new_vehicles = len(new_ids)
    total_unique_vehicles = len(seen_vehicle_ids)

    # -------------------------
    # LOGGING
    # -------------------------
    print(
        f"[{frame_num}] "
        f"CURRENT: {current_vehicles} | "
        f"NEW: {new_vehicles} | "
        f"TOTAL: {total_unique_vehicles}"
    )

    # -------------------------
    # DEBUG SNAPSHOT (on vehicle detection)
    # -------------------------
    if current_vehicles > 0:
        annotated = draw_vehicle_detections(frame, vehicle_boxes, ids)
        snapshot_path = os.path.join(DEBUG_DIR, f"frame_{frame_num}.jpg")
        cv2.imwrite(snapshot_path, annotated)

    frame_num += 1
