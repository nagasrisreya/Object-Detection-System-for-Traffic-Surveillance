"""
YOLO Detection Engine for Traffic Surveillance
Handles object detection using YOLOv8 models
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch


class TrafficDetector:
    """
    Traffic Object Detection using YOLO models
    Detects vehicles, pedestrians, traffic signs and signals
    """
    
    # COCO class indices relevant to traffic
    TRAFFIC_CLASSES = {
        # Vehicles
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck',
        1: 'bicycle',
        # Person
        0: 'person',
    }
    
    # Extended detection with custom model would include:
    # Traffic signs: stop, yield, speed limit, etc.
    # Traffic signals: red light, green light, yellow light
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.25):
        """
        Initialize the detector
        
        Args:
            model_name: YOLO model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            confidence: Minimum confidence threshold for detections
        """
        self.model_name = model_name
        self.confidence = confidence
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = []
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.class_names = self.model.names
            print(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detection dictionaries with bbox, class, confidence
        """
        if self.model is None:
            self.load_model()
            
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections.append(detection)
                
        return detections
    
    def detect_with_traffic_signs(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects including traffic signs and signals
        Uses specialized detection for traffic elements
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        detections = self.detect(frame)
        
        # For traffic signs/signals detection, we use the same model
        # but could be extended with custom trained model
        # Add traffic sign detection if available in model
        
        return detections
    

    
    def get_statistics(self, detections: List[Dict]) -> Dict:
        """
        Get statistics from detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with count statistics
        """
        stats = {
            'total': len(detections),
            'by_class': {}
        }
        
        for det in detections:
            class_name = det['class_name']
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = 0
            stats['by_class'][class_name] += 1
            
        return stats




