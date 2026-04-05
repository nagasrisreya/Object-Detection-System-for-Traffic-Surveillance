"""
Simple Traffic Detection App
Image and Video Upload → YOLOv8 Detection of Vehicles, Signs & Signals
Input only, model detects all frames, outputs annotated + stats
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
import sys

# Import local modules
from detector import TrafficDetector
from utils import (
    draw_advanced_detections,
    get_traffic_element_name,
    save_uploaded_file,
    cleanup_temp_file,
)

# Page configuration
st.set_page_config(
    page_title="Traffic Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .detection-log {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        max-height: 300px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


class TrafficApp:
    """Main application class"""
    
    def __init__(self):
        self.detector = None
        self.config = None
        self.temp_files = []
        
    def initialize_detector(self):
        """Initialize the YOLO detector"""
        if self.detector is None:
            with st.spinner('Loading YOLO model...'):
                self.detector = TrafficDetector(
                    model_name=self.config['model'],
                    confidence=self.config['confidence']
                )
                self.detector.load_model()
        return self.detector
    
    def process_webcam(self):
        """Process webcam feed"""
        st.subheader("📹 Webcam Feed")
        
        # Camera input type selection
        camera_type = st.radio(
            "Camera Type",
            ["Local Camera", "IP Camera (HTTP/HTTPS)"],
            horizontal=True,
            help="Select local camera or IP camera stream"
        )
        
        if camera_type == "Local Camera":
            # Local webcam selection
            camera_id = st.selectbox(
                "Select Camera",
                options=[0, 1, 2, 3],
                format_func=lambda x: f"Camera {x}" if x > 0 else "Default Camera"
            )
            video_source = camera_id
        else:
            # IP Camera URL input - select preset first
            preset = st.selectbox(
                "Camera Preset",
                ["IP Webcam (Android)", "Generic MJPEG", "Custom URL"]
            )
            
            video_source = ""
            
            if preset == "IP Webcam (Android)":
                # IP Webcam (Android) app - user enters the full URL shown in app
                video_source = st.text_input(
                    "Enter IP Webcam URL",
                    placeholder="http://192.168.1.100:8080/video",
                    help="Enter the exact URL from your IP Webcam app"
                )
            elif preset == "Generic MJPEG":
                col1, col2 = st.columns(2)
                with col1:
                    ip_address = st.text_input("Camera IP", placeholder="192.168.1.100")
                with col2:
                    port = st.text_input("Port", value="8080")
                if ip_address and port:
                    video_source = f"http://{ip_address}:{port}/video"
            elif preset == "Custom URL":
                video_source = st.text_input(
                    "Enter Camera URL",
                    placeholder="http://192.168.1.100:8080/video"
                )
        
        run_webcam = st.checkbox("Start Camera Detection", value=False)
        
        if run_webcam:
            # Determine video source
            if isinstance(video_source, str) and (video_source.startswith('http://') or video_source.startswith('https://')):
                # URL-based stream
                cap = cv2.VideoCapture(video_source)
                source_type = "IP Camera"
            else:
                # Local camera
                cap = cv2.VideoCapture(int(video_source) if isinstance(video_source, str) and video_source.isdigit() else video_source)
                source_type = f"Camera {video_source}"
            
            if not cap.isOpened():
                st.error(f"Could not open {source_type}")
                return
            
            st.success(f"{source_type} connected successfully!")
            
            # Create placeholder for video
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            frame_count = 0
            fps = 0
            
            try:
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame from camera")
                        break
                    
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    # Draw detections
                    annotated_frame = draw_advanced_detections(
                        frame, detections,
                        show_labels=self.config['show_labels'],
                        show_conf=self.config['show_conf']
                    )
                    
                    # Convert to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    video_placeholder.image(annotated_rgb, caption="Real-time Detection", use_container_width=True)
                    
                    # Update statistics
                    stats = self.detector.get_statistics(detections)
                    stats_placeholder.json(stats)
                    
                    frame_count += 1
                    
                    # FPS calculation
                    if frame_count % 30 == 0:
                        pass
                    
                    # Control frame rate
                    time.sleep(1.0 / self.config['stream_fps'])
                    
            finally:
                cap.release()
                video_placeholder.empty()
                stats_placeholder.empty()
                
    def process_video_file(self, uploaded_file):
        """Process uploaded video file"""
        st.subheader("🎬 Video File Processing")
        
        # Save uploaded file
        video_path = save_uploaded_file(uploaded_file)
        if not video_path:
            st.error("Failed to save uploaded video")
            return
            
        self.temp_files.append(video_path)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Failed to open video file")
            return
            
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.video(uploaded_file)
        
        st.info(f"Video Info: {width}x{height} | {fps} FPS | {total_frames} frames")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            skip_frames = st.slider("Skip Frames", 1, 10, 1, help="Process every N frames for speed")
        with col2:
            max_frames = st.slider("Max Frames to Process", 10, total_frames, min(300, total_frames))
        
        process_btn = st.button("🚀 Start Processing", type="primary")
        
        if process_btn:
            # Create placeholders
            video_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            frame_count = 0
            all_stats = []
            
            try:
                while cap.isOpened() and frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Skip frames for speed
                    for _ in range(skip_frames - 1):
                        ret, frame = cap.read()
                        if not ret:
                            break
                    
                    if not ret:
                        break
                    
                    # Detect objects
                    detections = self.detector.detect(frame)
                    
                    # Draw detections
                    annotated_frame = draw_advanced_detections(
                        frame, detections,
                        show_labels=self.config['show_labels'],
                        show_conf=self.config['show_conf']
                    )
                    
                    # Convert to RGB
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display
                    video_placeholder.image(annotated_rgb, caption=f"Frame {frame_count + 1}", use_container_width=True)
                    
                    # Statistics
                    stats = self.detector.get_statistics(detections)
                    all_stats.append(stats)
                    stats_placeholder.json(stats)
                    
                    # Progress
                    progress_bar.progress((frame_count + 1) / max_frames)
                    
                    frame_count += 1
                    
                    # Control speed
                    time.sleep(1.0 / self.config['stream_fps'])
                    
                # Final summary
                if all_stats:
                    st.success(f"Processed {frame_count} frames!")
                    
                    # Aggregate statistics
                    total_detections = sum(s['total'] for s in all_stats)
                    st.metric("Total Detections", total_detections)
                    
                    # By class summary
                    class_counts = {}
                    for s in all_stats:
                        for cls, count in s['by_class'].items():
                            class_counts[cls] = class_counts.get(cls, 0) + count
                    
                    if class_counts:
                        st.write("### Detection Summary")
                        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                            name = get_traffic_element_name(cls)
                            st.write(f"- {name}: {count}")
                            
            finally:
                cap.release()
                progress_bar.empty()
                
    def process_rtsp_stream(self):
        """Process RTSP/IP Camera stream"""
        st.subheader("🌐 RTSP/IP Camera Stream")
        
        # RTSP URL input
        rtsp_url = st.text_input(
            "Enter RTSP URL",
            placeholder="rtsp://username:password@ip_address:port/stream",
            help="Enter the RTSP URL for your IP camera"
        )
        
        # Quick presets
        preset = st.selectbox(
            "Quick Presets",
            ["Custom URL", "Localhost Test", "Example Camera"]
        )
        
        if preset == "Localhost Test":
            rtsp_url = "rtsp://localhost:8554/live"
        elif preset == "Example Camera":
            rtsp_url = "rtsp://example.com/stream"
            
        if rtsp_url:
            if not validate_rtsp_url(rtsp_url):
                st.warning("URL should start with rtsp:// or rtsps://")
                
            connect_btn = st.button("🔗 Connect to Camera", type="primary")
            
            if connect_btn:
                cap = cv2.VideoCapture(rtsp_url)
                
                if not cap.isOpened():
                    st.error("Failed to connect to RTSP stream. Check the URL and network connection.")
                    return
                    
                st.success("Connected to RTSP stream!")
                
                # Create placeholder
                video_placeholder = st.empty()
                stats_placeholder = st.empty()
                
                frame_count = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Lost connection to stream")
                            break
                            
                        # Detect objects
                        detections = self.detector.detect(frame)
                        
                        # Draw detections
                        annotated_frame = draw_advanced_detections(
                            frame, detections,
                            show_labels=self.config['show_labels'],
                            show_conf=self.config['show_conf']
                        )
                        
                        # Convert to RGB
                        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display
                        video_placeholder.image(annotated_rgb, caption=f"RTSP Stream - Frame {frame_count}", use_container_width=True)
                        
                        # Statistics
                        stats = self.detector.get_statistics(detections)
                        stats_placeholder.json(stats)
                        
                        frame_count += 1
                        
                        # Control frame rate
                        time.sleep(1.0 / self.config['stream_fps'])
                        
                except Exception as e:
                    st.error(f"Stream error: {e}")
                finally:
                    cap.release()
                    video_placeholder.empty()
                    stats_placeholder.empty()
                    
    def run(self):
        """Run the main application"""
        
        # Header
        st.title("🚗 Traffic Vision")
        st.markdown("### Real-Time Object Detection for Traffic Surveillance")
        st.markdown("---")
        
        # Simplified config
        self.config = {
            'model': 'yolov8n.pt',
            'confidence': 0.25,
            'show_labels': True,
            'show_conf': True,
            'stream_fps': 15
        }
        
        # Initialize detector
        self.initialize_detector()
        
        # Input source selection

        
        # Video processing (main focus)
        uploaded_file = st.file_uploader(
            "Upload Video (mp4/avi/mov)",
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        if uploaded_file:
            self.process_video_file(uploaded_file)
            
        # Cleanup temp files
        for f in self.temp_files:
            cleanup_temp_file(f)
            
        # Information section
        st.markdown("---")
        st.markdown("""
        ### 📋 Detection Classes
        The system detects the following traffic elements:
        
        **Vehicles:**
        - 🚗 Cars
        - 🏍️ Motorcycles  
        - 🚌 Buses
        - 🚚 Trucks
        - 🚲 Bicycles
        
        **Pedestrians:**
        - 👤 Persons
        
        **Traffic Signs & Signals:**
        - 🛑 Stop Signs
        - 🚦 Traffic Lights
        - ⚠️ Warning Signs
        - � Speed Limit Signs
        """)
        
        # Model info
        st.markdown("---")
        st.caption(f"Powered by YOLOv8 | Model: {self.config['model']} | Device: {self.detector.device if self.detector else 'N/A'}")


def main():
    """Main entry point"""
    try:
        app = TrafficApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

