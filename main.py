import os
import time
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from utils.tracking import update_tracker
from utils.lanes import get_lane, draw_lanes

# --- Video file ---
video_file = "traffic_demo.mp4"

if not os.path.exists(video_file):
    print(f"Video '{video_file}' not found. Please place it in the folder.")
    exit(1)
else:
    print(f"Using existing video '{video_file}'.")

# --- Load YOLOv8 model ---
model = YOLO("yolov8n.pt")
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# --- Start video capture ---
cap = cv2.VideoCapture(video_file)
vehicle_log = []
counted_ids = set()
lane_counts = {1: 0, 2: 0, 3: 0}
frame_count = 0

# --- Video writer with width/height safety ---
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('traffic_output.mp4', fourcc, 30, (width, height))

# --- Processing loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = time.strftime('%H:%M:%S', time.gmtime(cap.get(cv2.CAP_PROP_POS_MSEC)/1000))

    # Detect vehicles
    results = model(frame)[0]
    detections = []
    for r in results.boxes.data:
        x1, y1, x2, y2, score, class_id = r
        label = model.names[int(class_id)]
        # Lowered threshold to 0.2 to catch more vehicles
        if label in vehicle_classes and score > 0.2:
            detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])

    # Debug: print detections per frame
    print(f"Frame {frame_count}: {len(detections)} vehicles detected")

    # Track vehicles
    tracked = update_tracker(np.array(detections))
    print(f"Tracked vehicles: {tracked}")

    for t in tracked:
        x1, y1, x2, y2, track_id = map(int, t)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        lane = get_lane(cx, cy)

        if lane is None:
            print(f"Vehicle {track_id} at ({cx},{cy}) not assigned to any lane")

        if track_id not in counted_ids and lane:
            counted_ids.add(track_id)
            lane_counts[lane] += 1
            vehicle_log.append([track_id, lane, frame_count, timestamp])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID:{track_id} L:{lane}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    draw_lanes(frame)
    for lane_id in lane_counts:
        cv2.putText(frame, f"Lane {lane_id}: {lane_counts[lane_id]}", (20, 40*lane_id),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Export to CSV ---
if vehicle_log:
    df = pd.DataFrame(vehicle_log, columns=["Vehicle ID", "Lane", "Frame", "Timestamp"])
    df.to_csv("vehicle_data.csv", index=False)
    print("CSV saved as 'vehicle_data.csv'.")
else:
    print("No vehicles detected. CSV will be empty.")

# --- Final summary ---
print("Processed video saved as 'traffic_output.mp4'.")
print("Total vehicles logged:", len(vehicle_log))
print("Lane counts summary:", lane_counts)
