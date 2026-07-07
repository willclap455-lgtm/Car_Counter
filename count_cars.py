import os
import cv2
import subprocess
import numpy as np
import torch
from rfdetr import RFDETRMedium

# -----------------------------
# CONFIG
# -----------------------------
RTSP_URL = "rtsp://admin:clancy252629@192.168.105.120:554/cam/realmonitor?channel=1&subtype=2"

# FFmpeg Frame Dimension Config (Match this to your model's native input size)
WIDTH, HEIGHT = 640, 640  
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus"}
DEBUG_DIR = "debug"

MAX_DIST = 100        # pixel distance threshold for track matching
CONFIRM_FRAMES = 6    # Updated per request: 3 consecutive frames
DEDUPE_IOU = 0.35     # Updated per request: aggressive overlap suppression

# -----------------------------
# TRACKER STATE
# -----------------------------
next_id = 0
tracks = {}       # id -> centroid
track_hits = {}   # id -> consecutive frames seen
seen_vehicle_ids = set()


def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def dedupe_vehicle_boxes(boxes, confidences=None, iou_thresh=DEDUPE_IOU):
    if len(boxes) == 0:
        return boxes

    indices = list(range(len(boxes)))
    if confidences is not None:
        indices.sort(key=lambda i: confidences[i], reverse=True)

    kept = []
    for i in indices:
        if any(box_iou(boxes[i], boxes[j]) > iou_thresh for j in kept):
            continue
        kept.append(i)

    return boxes[kept]


def match_or_create_tracks(centroids):
    global next_id, tracks

    if not centroids:
        return []

    track_ids = list(tracks.keys())
    candidate_pairs = []

    for ci, centroid in enumerate(centroids):
        for tid in track_ids:
            dist = np.linalg.norm(np.array(centroid) - np.array(tracks[tid]))
            if dist < MAX_DIST:
                candidate_pairs.append((dist, ci, tid))

    candidate_pairs.sort()

    assigned_centroid = {}
    used_tracks = set()

    for _, ci, tid in candidate_pairs:
        if ci in assigned_centroid or tid in used_tracks:
            continue
        assigned_centroid[ci] = tid
        used_tracks.add(tid)

    assigned_ids = []
    for ci, centroid in enumerate(centroids):
        if ci in assigned_centroid:
            tid = assigned_centroid[ci]
        else:
            tid = next_id
            next_id += 1

        tracks[tid] = centroid
        assigned_ids.append(tid)

    return assigned_ids


def update_track_lifecycle(ids):
    active_ids = set(ids)

    for tid in ids:
        track_hits[tid] = track_hits.get(tid, 0) + 1

    for tid in list(track_hits.keys()):
        if tid not in active_ids:
            track_hits[tid] = 0


def confirm_new_vehicle_ids(ids):
    newly_confirmed = []

    for tid in ids:
        if track_hits[tid] == CONFIRM_FRAMES and tid not in seen_vehicle_ids:
            seen_vehicle_ids.add(tid)
            newly_confirmed.append(tid)

    return newly_confirmed


def draw_vehicle_detections(frame, boxes, ids, highlight_ids=None):
    annotated = frame.copy()
    highlight_ids = highlight_ids or set()

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = get_centroid(box)
        cx, cy = int(cx), int(cy)

        is_new = track_id in highlight_ids
        box_color = (0, 0, 255) if is_new else (0, 255, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
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
            box_color,
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
# FFmpeg RTSP SUBPROCESS
# -----------------------------
print("Launching FFmpeg RTSP pipeline...")
ffmpeg_cmd = [
    'ffmpeg',
    '-rtsp_transport', 'tcp',                 # Prevents UDP frame drops over network
    '-i', RTSP_URL,
    '-vf', f'fps=8,scale={WIDTH}:{HEIGHT}',   # Throttles to 4fps and resizes frame in C
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',                      # Uses bgr24 directly so cv2 can save natively
    '-vcodec', 'rawvideo',
    '-'
]

process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=FRAME_SIZE)
frame_num = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        # Pull raw pre-scaled frame bytes out of the pipeline
        in_bytes = process.stdout.read(FRAME_SIZE)
        if len(in_bytes) != FRAME_SIZE:
            print("RTSP Stream broken or ended.")
            break

        # Reconstruct into frame array
        frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))

        # -------------------------
        # INFERENCE
        # -------------------------
        detections = model.predict(frame, threshold=0.5)

        class_names = detections.data["class_name"]
        xyxy = detections.xyxy

        vehicle_mask = np.isin(class_names, list(VEHICLE_CLASSES))
        vehicle_boxes = xyxy[vehicle_mask]

        vehicle_confidences = None
        if hasattr(detections, "confidence") and detections.confidence is not None:
            vehicle_confidences = detections.confidence[vehicle_mask]

        vehicle_boxes = dedupe_vehicle_boxes(vehicle_boxes, vehicle_confidences)
        centroids = [get_centroid(b) for b in vehicle_boxes]

        # -------------------------
        # TRACKING
        # -------------------------
        ids = match_or_create_tracks(centroids)
        update_track_lifecycle(ids)
        newly_confirmed = confirm_new_vehicle_ids(ids)

        current_vehicles = len(ids)
        new_vehicles = len(newly_confirmed)
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
        # DEBUG SNAPSHOT
        # -------------------------
        if newly_confirmed:
            annotated = draw_vehicle_detections(
                frame, vehicle_boxes, ids, highlight_ids=set(newly_confirmed)
            )
            snapshot_path = os.path.join(DEBUG_DIR, f"frame_{frame_num}.jpg")
            cv2.imwrite(snapshot_path, annotated)

            # Optional: Clean up old coordinates to stop infinite memory growth
            # on dead tracking points out of frame
            if len(tracks) > 100:
                tracks = {tid: tracks[tid] for tid in ids}

        frame_num += 1

finally:
    process.stdout.close()
    process.terminate()
