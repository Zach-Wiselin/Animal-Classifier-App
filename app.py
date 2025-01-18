import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Load the MobileNet model
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.models.load_model(model_url, compile=False)
    ])
    return model

model = load_model()

# Load labels for MobileNet
@st.cache_data
def load_labels():
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    response = requests.get(labels_url)
    response.raise_for_status()
    labels = response.text.splitlines()
    return labels

labels = load_labels()

# Streamlit App
st.title("Lightweight Animal Classifier 🐾")
st.write("Capture an image using your webcam to classify animals efficiently!")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Process the captured image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess the image for MobileNet
    input_image = image.resize((224, 224))  # Resize to 224x224 pixels
    input_array = np.array(input_image, dtype=np.float32)[np.newaxis, ...]  # Add batch dimension

    # Perform classification
    predictions = model.predict(input_array)
    predicted_label = labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.write(f"### Detected: {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Fetch additional information about the detected label
    st.write("Fetching more information...")
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{predicted_label}")
        if response.status_code == 200:
            data = response.json()
            animal_info = data.get("extract", "No additional information available.")
        else:
            animal_info = "No additional information available."
    except Exception:
        animal_info = "Failed to fetch additional information."

    st.write(f"**Info about {predicted_label}:** {animal_info}")

# Footer
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ using TensorFlow Lite</div>", unsafe_allow_html=True)
