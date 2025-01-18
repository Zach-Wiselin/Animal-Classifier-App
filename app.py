import streamlit as st
from PIL import Image
import cv2
import torch
import numpy as np
from pathlib import Path
import requests
import subprocess

# Clone YOLOv5 dynamically if not present
MODEL_PATH = Path("yolov5")
if not MODEL_PATH.exists():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(MODEL_PATH)])

# Load YOLOv5 model (using a fine-tuned or pre-trained weights path)
model = torch.hub.load(str(MODEL_PATH), 'yolov5s', source='local', pretrained=True)

# Expanded animal and bird classes (subset of COCO + others)
animal_classes = [
    "cat", "dog", "bird", "cow", "horse", "sheep", "elephant", "bear", "zebra", "giraffe", "lion", "tiger",
    "deer", "fox", "rabbit", "kangaroo", "leopard", "wolf", "monkey", "panda", "peacock", "eagle", "owl",
    "parrot", "penguin", "sparrow", "falcon", "flamingo", "dove", "hawk", "woodpecker"
]

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.6  # Filter out predictions below this confidence

# Streamlit app
st.title("Animal Classifier App 🐾")

st.write("Use your webcam feed to classify animals or birds in real-time with improved accuracy!")

# Centered Start Webcam Checkbox
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
run = st.checkbox('Start Webcam')
st.markdown("</div>", unsafe_allow_html=True)

FRAME_WINDOW = st.empty()
info_placeholder = st.empty()
cap = cv2.VideoCapture(0)

captured_animal = None

# Centered Retry Button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
retry = st.button("Retry")
st.markdown("</div>", unsafe_allow_html=True)

if retry:
    captured_animal = None

if run:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(frame_rgb)
        detections = results.pandas().xyxy[0]  # Get detection results

        for index, detection in detections.iterrows():
            label = detection['name']
            confidence = detection['confidence']

            # Only process animal classes with sufficient confidence
            if label in animal_classes and confidence > CONFIDENCE_THRESHOLD:
                # Draw bounding box
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label.upper()} ({confidence * 100:.2f}%)"
                cv2.putText(frame_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if captured_animal is None:
                    captured_animal = label
                    st.success(f"Captured: {captured_animal.upper()}!")

        # Display the frame with bounding boxes
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

        # Populate animal information
        if captured_animal:
            animal_info = f"Loading information for {captured_animal}..."
            info_placeholder.markdown(f"### Animal Information:")

            # Fetch information from an API (e.g., Wikipedia API)
            try:
                response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{captured_animal}")
                if response.status_code == 200:
                    data = response.json()
                    animal_info = data.get("extract", "No information available.")
            except:
                animal_info = "Failed to retrieve information."

            # Display the information
            info_placeholder.markdown(
                f"<div style='text-align: center;'>"
                f"<h2>It's a {captured_animal.capitalize()}!</h2>"
                f"<p>{animal_info}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            break
else:
    st.write("Check the box above to start the webcam feed.")

# Footer with heart emoji
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ by Zach</div>", unsafe_allow_html=True)

# Release the webcam
cap.release()
