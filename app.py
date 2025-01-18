import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
from pathlib import Path
import subprocess
import requests

# Ensure YOLOv5 is available
MODEL_PATH = Path("yolov5")
if not MODEL_PATH.exists():
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git", str(MODEL_PATH)])
    subprocess.run(["pip", "install", "-r", str(MODEL_PATH / "requirements.txt")])

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load(str(MODEL_PATH), "yolov5s", source="local", pretrained=True)

model = load_model()

# Define animal classes
animal_classes = [
    "cat", "dog", "bird", "cow", "horse", "sheep", "elephant", "bear", "zebra", "giraffe", "lion", "tiger",
    "deer", "fox", "rabbit", "kangaroo", "leopard", "wolf", "monkey", "panda", "peacock", "eagle", "owl",
    "parrot", "penguin", "sparrow", "falcon", "flamingo", "dove", "hawk", "woodpecker"
]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.6

# Streamlit app
st.title("Animal Classifier App 🐾")
st.write("Use your webcam to classify animals or birds in real-time!")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Process the image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Perform object detection
    results = model(img_array)
    detections = results.pandas().xyxy[0]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for _, detection in detections.iterrows():
        label = detection["name"]
        confidence = detection["confidence"]

        # Only process animal classes with sufficient confidence
        if label in animal_classes and confidence > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, [detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"]])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), f"{label.upper()} ({confidence * 100:.2f}%)", fill="green")

    # Display the image with bounding boxes
    st.image(image, caption="Processed Image", use_column_width=True)

    # Fetch additional information about detected animals
    st.write("Animal Information:")
    for _, detection in detections.iterrows():
        label = detection["name"]
        if label in animal_classes:
            try:
                response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{label}")
                if response.status_code == 200:
                    data = response.json()
                    animal_info = data.get("extract", "No additional information available.")
                else:
                    animal_info = "No additional information available."
            except Exception:
                animal_info = "Failed to fetch additional information."
            st.write(f"**{label.capitalize()}:** {animal_info}")

# Footer
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ using YOLOv5</div>", unsafe_allow_html=True)
