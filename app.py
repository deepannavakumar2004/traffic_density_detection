import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Load the YOLOv8 model
best_model = YOLO('yolov8n.pt')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    heavy_traffic_threshold = 10
    lane_threshold = 609
    
    stframe = st.empty()  # Create a Streamlit frame to display video
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detection_frame = frame.copy()
        results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        processed_frame = results[0].plot(line_width=1)
        
        bounding_boxes = results[0].boxes
        vehicles_in_left_lane = sum(1 for box in bounding_boxes.xyxy if box[0] < lane_threshold)
        vehicles_in_right_lane = sum(1 for box in bounding_boxes.xyxy if box[0] >= lane_threshold)
        
        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"
        
        cv2.putText(processed_frame, f'Left Lane: {traffic_intensity_left}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed_frame, f'Right Lane: {traffic_intensity_right}', (820, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert frame to RGB for Streamlit
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the processed frame in Streamlit
        stframe.image(processed_frame, channels="RGB")
    
    cap.release()

st.title("Traffic Detection App")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name
    
    if st.button("Process Video"):
        process_video(temp_video_path)
