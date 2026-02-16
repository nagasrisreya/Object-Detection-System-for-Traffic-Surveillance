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
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_labels: bool = True, show_conf: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            show_labels: Whether to show class labels
            show_conf: Whether to show confidence scores
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        # Color map for different classes
        color_map = {
            'car': (0, 255, 0),           # Green
            'motorcycle': (255, 0, 255),   # Magenta
            'bus': (0, 165, 255),          # Orange
            'truck': (0, 255, 255),        # Yellow
            'bicycle': (255, 255, 0),      # Cyan
            'person': (255, 0, 0),        # Blue
            'traffic sign': (128, 0, 128), # Purple
            'traffic light': (0, 128, 128), # Teal
            'stop sign': (0, 0, 255),      # Red
            'speed limit': (128, 0, 0),    # Maroon
        }
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for this class
            color = color_map.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if show_labels and show_conf:
                label = f"{class_name}: {confidence:.2f}"
            elif show_labels:
                label = class_name
            else:
                label = f"{confidence:.2f}"
                
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return output
    
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


class VideoProcessor:
    """Process video streams and files"""
    
    def __init__(self, detector: TrafficDetector):
        self.detector = detector
        
    def process_video_file(self, video_path: str, 
                          output_path: Optional[str] = None,
                          show_display: bool = False) -> Dict:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_display: Whether to show real-time display
            
        Returns:
            Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': 'Failed to open video file'}
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        frame_count = 0
        all_stats = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Draw detections
            annotated_frame = self.detector.draw_detections(frame, detections)
            
            # Get statistics
            stats = self.detector.get_statistics(detections)
            stats['frame'] = frame_count
            all_stats.append(stats)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
                
            # Display frame
            if show_display:
                cv2.imshow('Traffic Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
                
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()
            
        return {
            'frames_processed': frame_count,
            'statistics': all_stats
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated frame, detections)
        """
        detections = self.detector.detect(frame)
        annotated_frame = self.detector.draw_detections(frame, detections)
        stats = self.detector.get_statistics(detections)
        
        return annotated_frame, detections, stats

