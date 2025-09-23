"""
Live Vehicle Counting with YOLOv8
Processes a video stream and outputs live vehicle counts for dashboard or RL environment.
"""

import cv2
import numpy as np
import time
from src.cv.vehicle_detector import VehicleDetector

# Path to YOLOv8 weights (update as needed)
MODEL_PATH = "yolov8n.pt"
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
OUTPUT_PATH = "logs/live_vehicle_counts.txt"

# Optional: Define lane regions (list of polygons)
lane_regions = []  # Fill with np.ndarray polygons if available


def main():
    detector = VehicleDetector(model_path=MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {VIDEO_SOURCE}")
        return

    with open(OUTPUT_PATH, "w") as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Detect vehicles
            detections = detector.detect_vehicles(frame)
            vehicle_count = len(detections)
            # Optionally count per lane
            if lane_regions:
                lane_counts = detector.count_vehicles_by_lane(frame, lane_regions)
            else:
                lane_counts = None
            # Write results to file (timestamp, total count, lane counts)
            ts = time.time()
            f.write(f"{ts},{vehicle_count},{lane_counts}\n")
            f.flush()
            # Display (optional)
            cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Live Vehicle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
