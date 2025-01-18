import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests

# Load the TensorFlow Hub model
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    return hub.KerasLayer(model_url)

model = load_model()

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input size
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Fetch ImageNet labels
@st.cache_data
def fetch_imagenet_labels():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    response = requests.get(url)
    return response.text.splitlines()

imagenet_labels = fetch_imagenet_labels()

# Streamlit App
st.markdown(
    """
    <style>
    .center {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        font-size: 18px;
        margin-bottom: 20px;
        color: #666;
    }
    .footer {
        margin-top: 50px;
        font-size: 14px;
        color: #888;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='center title'>Animal Classifier 🐾</div>", unsafe_allow_html=True)
st.markdown("<div class='center subtitle'>Capture an image using your webcam to classify animals.</div>", unsafe_allow_html=True)

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    image = Image.open(camera_input).convert("RGB")
    st.markdown("<div class='center'>", unsafe_allow_html=True)
    st.image(image, caption="Captured Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Preprocess the image and make predictions
    input_array = preprocess_image(image)
    predictions = model(input_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_label = imagenet_labels[predicted_class_idx]

    # Display results
    st.markdown(
        f"<div class='center'><h3>It's a <b>{predicted_label.capitalize()}</b>!</h3></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='center'><b>Confidence:</b> {confidence:.2f}%</div>", unsafe_allow_html=True)

    # Fetch animal info from Wikipedia
    st.markdown(f"<div class='center'>Fetching information about <b>{predicted_label}</b>...</div>", unsafe_allow_html=True)
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{predicted_label}")
        if response.status_code == 200:
            data = response.json()
            animal_info = data.get("extract", "No additional information available.")
        else:
            animal_info = "No additional information available."
    except Exception:
        animal_info = "Failed to fetch additional information."
    st.markdown(f"<div class='center'><b>Info about {predicted_label}:</b> {animal_info}</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='center footer'>Made with ❤️ by Zach</div>", unsafe_allow_html=True)
