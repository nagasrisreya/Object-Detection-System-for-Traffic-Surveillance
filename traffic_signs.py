"""
Traffic Sign and Signal Detection Module
Specialized detection for traffic signs and traffic lights
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class TrafficSignDetector:
    """
    Specialized detector for traffic signs and signals
    Uses color-based and shape-based detection as fallback
    """
    
    # HSV color ranges for traffic elements
    COLOR_RANGES = {
        'red': [
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255))
        ],
        'yellow': [
            ((15, 100, 100), (35, 255, 255))
        ],
        'green': [
            ((40, 50, 50), (85, 255, 255))
        ],
        'blue': [
            ((90, 100, 100), (130, 255, 255))
        ],
        'white': [
            ((0, 0, 200), (180, 50, 255))
        ]
    }
    
    # Traffic sign shapes
    SHAPES = {
        'octagon': 8,      # Stop sign
        'triangle': 3,      # Warning signs
        'rectangle': 4,     # Speed limit, informational
        'circle': -1       # Most regulatory signs
    }
    
    # Traffic sign templates/descriptions
    TRAFFIC_SIGN_INFO = {
        'stop': {
            'name': 'Stop Sign',
            'description': 'Stop completely',
            'color': 'red',
            'shape': 'octagon'
        },
        'yield': {
            'name': 'Yield Sign',
            'description': 'Yield to other vehicles',
            'color': 'red',
            'shape': 'triangle'
        },
        'speed_limit': {
            'name': 'Speed Limit',
            'description': 'Maximum speed allowed',
            'color': 'white',
            'shape': 'rectangle'
        },
        'no_entry': {
            'name': 'No Entry',
            'description': 'No vehicles allowed',
            'color': 'red',
            'shape': 'circle'
        },
        'red_light': {
            'name': 'Red Light',
            'description': 'Stop - Red traffic light',
            'color': 'red',
            'shape': 'circle'
        },
        'green_light': {
            'name': 'Green Light',
            'description': 'Go - Green traffic light',
            'color': 'green',
            'shape': 'circle'
        },
        'yellow_light': {
            'name': 'Yellow Light',
            'description': 'Slow down - Yellow light',
            'color': 'yellow',
            'shape': 'circle'
        },
        'pedestrian': {
            'name': 'Pedestrian Crossing',
            'description': 'Pedestrian crossing ahead',
            'color': 'blue',
            'shape': 'rectangle'
        }
    }
    
    def __init__(self):
        """Initialize the traffic sign detector"""
        self.min_area = 500
        self.max_area = 50000
        
    def detect_by_color(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect traffic elements by color
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected traffic elements
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        
        for color_name, ranges in self.COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                lower_np = np.array(lower)
                upper_np = np.array(upper)
                mask |= cv2.inRange(hsv, lower_np, upper_np)
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Determine sign type based on color
                    sign_type = self._infer_sign_type(color_name)
                    
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'color': color_name,
                        'type': sign_type,
                        'name': self.TRAFFIC_SIGN_INFO.get(sign_type, {}).get('name', 'Unknown'),
                        'confidence': 0.7
                    })
                    
        return detections
    
    def _infer_sign_type(self, color: str) -> str:
        """
        Infer traffic sign type from detected color
        
        Args:
            color: Detected color
            
        Returns:
            Sign type string
        """
        mapping = {
            'red': 'stop',
            'yellow': 'yellow_light',
            'green': 'green_light',
            'blue': 'pedestrian',
            'white': 'speed_limit'
        }
        return mapping.get(color, 'unknown')
    
    def detect_by_shape(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect traffic signs by shape
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected shapes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # Approximate shape
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Determine shape
                shape = self._identify_shape(approx)
                
                if shape:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'shape': shape,
                        'confidence': 0.6
                    })
                    
        return detections
    
    def _identify_shape(self, approx: np.ndarray) -> Optional[str]:
        """
        Identify shape from polygon approximation
        
        Args:
            approx: Approximated polygon
            
        Returns:
            Shape name or None
        """
        vertices = len(approx)
        
        if vertices == 8:
            return 'octagon'
        elif vertices == 3:
            return 'triangle'
        elif vertices == 4:
            return 'rectangle'
        elif vertices > 6:
            return 'circle'
            
        return None
    
    def draw_traffic_sign_labels(self, frame: np.ndarray, 
                                 detections: List[Dict]) -> np.ndarray:
        """
        Draw labels for traffic signs with recognition names
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Get color based on sign type
            color_map = {
                'stop': (0, 0, 255),
                'red_light': (0, 0, 255),
                'yellow_light': (0, 255, 255),
                'green_light': (0, 255, 0),
                'speed_limit': (255, 255, 255),
                'pedestrian': (255, 0, 0)
            }
            
            sign_type = det.get('type', 'unknown')
            color = color_map.get(sign_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Get display name
            display_name = det.get('name', 'Traffic Sign')
            
            # Draw label
            label = f"{display_name}"
            if 'confidence' in det:
                label += f" ({det['confidence']*100:.0f}%)"
                
            # Label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, (x1, y1 - label_h - 10),
                         (x1 + label_w, y1), color, -1)
            
            # Label text
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return output


def combine_detections(yolo_detections: List[Dict], 
                       sign_detections: List[Dict]) -> List[Dict]:
    """
    Combine YOLO detections with color/shape based detections
    
    Args:
        yolo_detections: Detections from YOLO model
        sign_detections: Detections from color/shape detection
        
    Returns:
        Combined and deduplicated detections
    """
    combined = list(yolo_detections)
    
    # Add sign detections that don't overlap with YOLO detections
    for sign_det in sign_detections:
        is_duplicate = False
        
        for yolo_det in yolo_detections:
            # Check for significant overlap
            if _compute_iou(sign_det['bbox'], yolo_det['bbox']) > 0.5:
                is_duplicate = True
                break
                
        if not is_duplicate:
            combined.append(sign_det)
            
    return combined


def _compute_iou(box1: List[int], box2: List[int]) -> float:
    """
    Compute Intersection over Union between two boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

