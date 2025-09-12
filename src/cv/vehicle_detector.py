"""
Vehicle Detection Module

This module provides vehicle detection and counting capabilities using
YOLOv8 and OpenCV for traffic analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class VehicleDetector:
    """
    Vehicle detection and counting system using YOLOv8.
    
    This class provides real-time vehicle detection, tracking, and counting
    capabilities for traffic analysis.
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Vehicle classes in COCO dataset
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected vehicles with bounding boxes and confidence scores
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a vehicle
                    class_id = int(box.cls[0])
                    if class_id in self.vehicle_classes:
                        detection = {
                            'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                            'confidence': float(box.conf[0]),
                            'class_id': class_id,
                            'class_name': self.model.names[class_id]
                        }
                        detections.append(detection)
        
        return detections
    
    def count_vehicles_by_lane(self, frame: np.ndarray, lane_regions: List[np.ndarray]) -> Dict[int, int]:
        """
        Count vehicles in specific lane regions.
        
        Args:
            frame: Input image frame
            lane_regions: List of lane region polygons
            
        Returns:
            Dictionary mapping lane_id to vehicle count
        """
        detections = self.detect_vehicles(frame)
        lane_counts = {}
        
        for lane_id, lane_region in enumerate(lane_regions):
            count = 0
            for detection in detections:
                # Check if vehicle center is within lane region
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                if cv2.pointPolygonTest(lane_region, (center_x, center_y), False) >= 0:
                    count += 1
            
            lane_counts[lane_id] = count
        
        return lane_counts
    
    def estimate_queue_length(self, frame: np.ndarray, stop_line_region: np.ndarray) -> int:
        """
        Estimate queue length at stop line.
        
        Args:
            frame: Input image frame
            stop_line_region: Polygon defining the stop line area
            
        Returns:
            Estimated number of vehicles in queue
        """
        detections = self.detect_vehicles(frame)
        queue_count = 0
        
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Check if vehicle is in stop line region
            if cv2.pointPolygonTest(stop_line_region, (center_x, center_y), False) >= 0:
                queue_count += 1
        
        return queue_count
