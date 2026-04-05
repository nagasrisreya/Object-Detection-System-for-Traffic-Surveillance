"""
Utility Functions for Traffic Surveillance System
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from pathlib import Path
import tempfile
import os


# Traffic Sign and Signal Names Mapping
TRAFFIC_SIGNS = {
    'stop': 'Stop Sign',
    'yield': 'Yield Sign', 
    'speed limit': 'Speed Limit Sign',
    'no entry': 'No Entry Sign',
    'parking': 'Parking Sign',
    'warning': 'Warning Sign',
    'traffic sign': 'Traffic Sign'
}

TRAFFIC_LIGHTS = {
    'red': 'Red Light',
    'green': 'Green Light',
    'yellow': 'Yellow Light',
    'traffic light': 'Traffic Light'
}


def get_traffic_element_name(class_name: str) -> str:
    """
    Get human-readable name for traffic elements
    
    Args:
        class_name: Internal class name
        
    Returns:
        Human-readable name
    """
    class_lower = class_name.lower()
    
    if 'stop' in class_lower:
        return TRAFFIC_SIGNS['stop']
    elif 'yield' in class_lower:
        return TRAFFIC_SIGNS['yield']
    elif 'speed' in class_lower:
        return TRAFFIC_SIGNS['speed limit']
    elif 'traffic' in class_lower and 'light' in class_lower:
        return TRAFFIC_LIGHTS['traffic light']
    elif 'traffic' in class_lower:
        return TRAFFIC_SIGNS['traffic sign']
    else:
        return class_name.title()


def create_color_map() -> Dict[str, Tuple[int, int, int]]:
    """
    Create color mapping for different object classes
    
    Returns:
        Dictionary of class names to BGR colors
    """
    return {
        # Vehicles
        'car': (0, 255, 0),           # Green
        'motorcycle': (255, 0, 255),   # Magenta
        'bus': (0, 165, 255),          # Orange
        'truck': (0, 255, 255),        # Yellow
        'bicycle': (255, 255, 0),      # Cyan
        'train': (128, 0, 128),        # Purple
        
        # Person
        'person': (255, 0, 0),         # Blue
        
        # Traffic Signs
        'traffic sign': (128, 0, 128), # Purple
        'stop sign': (0, 0, 255),       # Red
        'speed limit': (128, 0, 0),     # Maroon
        
        # Traffic Lights
        'traffic light': (0, 128, 128), # Teal
        'red light': (0, 0, 255),       # Red
        'green light': (0, 255, 0),     # Green
        'yellow light': (0, 255, 255),  # Yellow
        
        # Default
        'unknown': (128, 128, 128)     # Gray
    }


def draw_advanced_detections(frame: np.ndarray, detections: List[Dict],
                            show_labels: bool = True, 
                            show_conf: bool = True,
                            show_track: bool = False) -> np.ndarray:
    """
    Draw advanced detections with additional visual information
    
    Args:
        frame: Input frame
        detections: List of detection dictionaries
        show_labels: Show class labels
        show_conf: Show confidence scores
        show_track: Show tracking IDs
        
    Returns:
        Annotated frame
    """
    output = frame.copy()
    color_map = create_color_map()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color
        color = color_map.get(class_name, color_map['unknown'])
        
        # Draw filled rectangle for better visibility
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw semi-transparent overlay for label
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1 - 30), (x2, y1), color, -1)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        
        # Prepare label
        label_parts = []
        if show_labels:
            # Get readable name for traffic elements
            display_name = get_traffic_element_name(class_name)
            label_parts.append(display_name)
        if show_conf:
            label_parts.append(f"{confidence*100:.1f}%")
        if show_track and 'track_id' in det:
            label_parts.append(f"ID:{det['track_id']}")
            
        label = " ".join(label_parts)
        
        # Draw label
        cv2.putText(output, label, (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw corner markers for important detections
        if class_name in ['stop sign', 'traffic light', 'traffic sign']:
            draw_corner_markers(output, x1, y1, x2, y2, color)
            
    return output


def draw_corner_markers(frame: np.ndarray, x1: int, y1: int, 
                       x2: int, y2: int, color: Tuple[int, int, int],
                       length: int = 10, thickness: int = 2):
    """
    Draw corner markers on bounding box
    
    Args:
        frame: Frame to draw on
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        color: Line color
        length: Length of corner markers
        thickness: Line thickness
    """
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def save_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Save uploaded file to temporary directory
    
    Args:
        uploaded_file: Streamlit uploaded file
        
    Returns:
        Path to saved temporary file
    """
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        return tfile.name
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def cleanup_temp_file(filepath: str):
    """
    Remove temporary file
    
    Args:
        filepath: Path to file to remove
    """
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error removing file: {e}")


def create_statistics_display(stats: Dict) -> Dict:
    """
    Create display-friendly statistics
    
    Args:
        stats: Raw statistics dictionary
        
    Returns:
        Formatted statistics for display
    """
    display_stats = {
        'Total Objects': stats.get('total', 0),
        'Detection Summary': []
    }
    
    by_class = stats.get('by_class', {})
    for class_name, count in by_class.items():
        display_name = get_traffic_element_name(class_name)
        display_stats['Detection Summary'].append({
            'Object': display_name,
            'Count': count
        })
        
    return display_stats


def validate_rtsp_url(url: str) -> bool:
    """
    Validate RTSP URL format
    
    Args:
        url: RTSP URL to validate
        
    Returns:
        True if valid format
    """
    if not url:
        return False
    return url.startswith('rtsp://') or url.startswith('rtsps://')


def create_sample_rtsp_urls() -> List[Dict]:
    """
    Create sample RTSP URLs for testing
    
    Returns:
        List of sample URL configurations
    """
    return [
        {
            'name': 'Example IP Camera',
            'url': 'rtsp://username:password@ip_address:554/stream',
            'description': 'Replace with your IP camera credentials'
        },
        {
            'name': 'Local Camera (RTSP Simulator)',
            'url': 'rtsp://localhost:8554/live',
            'description': 'Requires RTSP server running locally'
        }
    ]


def format_detection_log(detections: List[Dict]) -> str:
    """
    Format detections for logging/display
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Formatted string
    """
    if not detections:
        return "No detections"
        
    lines = []
    for i, det in enumerate(detections, 1):
        class_name = get_traffic_element_name(det['class_name'])
        conf = det['confidence'] * 100
        bbox = det['bbox']
        lines.append(f"{i}. {class_name} ({conf:.1f}%) - Box: {bbox}")
        
    return "\n".join(lines)


# Streamlit UI helpers
def get_model_options() -> List[str]:
    """
    Get available YOLO model options
    
    Returns:
        List of model names with descriptions
    """
    return [
        ('yolov8n.pt', 'YOLOv8 Nano - Fastest, lowest accuracy'),
        ('yolov8s.pt', 'YOLOv8 Small - Fast, good accuracy'),
        ('yolov8m.pt', 'YOLOv8 Medium - Balanced'),
        ('yolov8l.pt', 'YOLOv8 Large - High accuracy'),
        ('yolov8x.pt', 'YOLOv8 XLarge - Highest accuracy, slowest'),
    ]


def create_sidebar_config() -> Dict:
    """
    Simplified sidebar for new app
    """
    return {}

