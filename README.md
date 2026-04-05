# Simple Traffic Detection App

## Project Overview
Simple Streamlit app for traffic detection.
Upload image or video → YOLOv8 detects vehicles, traffic signs & signals in all frames → annotated output + stats + JSON export.

## Features

### 1. Input Sources (User Selectable)
- **Webcam**: Live camera feed from connected webcam
- **Video File**: Upload and process pre-recorded video files
- **RTSP/IP Camera**: Stream from network cameras via RTSP protocol

### 2. Object Detection
- **Vehicles**: Cars, trucks, buses, motorcycles, bicycles
- **Pedestrians**: People walking, running
- **Traffic Signs**: Stop signs, speed limit signs, warning signs
- **Traffic Signals**: Red, yellow, green lights, pedestrian signals

### 3. Visualization
- Bounding boxes around detected objects
- Class labels with confidence scores
- Color-coded by object category
- Object counting statistics

### 4. User Interface
- Sidebar for configuration
- Real-time video display
- Detection statistics dashboard
- Detection history/log

## Technical Stack
- **Framework**: Streamlit (Web Application)
- **Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Processing**: CUDA (GPU) / CPU fallback

## Installation
```bash
pip install ultralytics opencv-python streamlit
```

## Usage
```bash
streamlit run app.py
```

## File Structure
- `app.py` - Main Streamlit application
- `detector.py` - YOLO detection engine
- `utils.py` - Utility functions
- `requirements.txt` - Dependencies

