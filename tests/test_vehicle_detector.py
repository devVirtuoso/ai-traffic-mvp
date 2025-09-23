"""
Unit tests for VehicleDetector (YOLOv8 CV module)
"""
import numpy as np
from src.cv.vehicle_detector import VehicleDetector

def test_detector_init():
    detector = VehicleDetector(model_path="yolov8n.pt")
    assert detector is not None

def test_detect_vehicles_empty():
    detector = VehicleDetector(model_path="yolov8n.pt")
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect_vehicles(blank_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0
