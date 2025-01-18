import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import requests
import tensorflow as tf

# Load the MobileNet model
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    return tf.keras.Sequential([tf.keras.layers.Input(shape=(224, 224, 3)), tf.keras.models.load_model(model_url)])

model = load_model()

# Load labels for MobileNet
@st.cache_data
def load_labels():
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    response = requests.get(labels_url)
    labels = response.text.splitlines()
    return labels

labels = load_labels()

# Streamlit App Title
st.title("Lightweight Animal Classifier 🐾")

st.write("Use your webcam or upload an image to classify animals efficiently!")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

# Image uploader (alternative input)
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# Process the image (from webcam or uploaded file)
if camera_input or uploaded_file:
    # Get the image
    image = Image.open(camera_input or uploaded_file).convert("RGB")

    # Preprocess the image for MobileNet
    input_image = image.resize((224, 224))  # Resize to MobileNet's expected input
    input_array = np.array(input_image) / 255.0  # Normalize pixel values
    input_array = np.expand_dims(input_array, axis=0)  # Add batch dimension

    # Perform classification
    predictions = model(input_array)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Draw results on the image
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), f"{predicted_label}: {confidence:.2f}%", fill="green")

    # Display the image with the label
    st.image(image, caption="Processed Image", use_column_width=True)
    st.write(f"### Detected: {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Display additional information about the detected animal
    st.write(f"Fetching more information about **{predicted_label}**...")
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{predicted_label}")
        if response.status_code == 200:
            data = response.json()
            animal_info = data.get("extract", "No additional information available.")
        else:
            animal_info = "No additional information available."
    except Exception:
        animal_info = "Failed to fetch additional information."

    st.write(f"**Info:** {animal_info}")

# Footer
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ for lightweight classification</div>", unsafe_allow_html=True)
