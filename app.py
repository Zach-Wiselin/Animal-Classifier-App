import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Load a pre-trained model for animal classification
@st.cache_resource
def load_animal_model():
    return tf.keras.models.load_model("https://tfhub.dev/google/aiy/vision/classifier/animals_V1/1")

model = load_animal_model()

# Preprocess image for the animal classifier model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to input size
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Class labels specific to the animal classification model
animal_labels = [
    "cat", "dog", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
    "lion", "tiger", "deer", "fox", "rabbit", "kangaroo", "leopard", "wolf", "monkey", 
    "panda", "peacock", "eagle", "owl", "parrot", "penguin", "sparrow", "falcon", 
    "flamingo", "dove", "hawk", "woodpecker"
]

# Streamlit App
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: gray;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Animal Classifier 🐾</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Capture an image using your webcam to classify animals.</div>', unsafe_allow_html=True)

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Process the captured image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    # Preprocess the image and predict
    input_array = preprocess_image(image)
    predictions = model.predict(input_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = animal_labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # Display results
    st.markdown(f"### It's a **{predicted_label}**!")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Fetch additional information about the detected label
    st.markdown(f"Fetching information about **{predicted_label}**...")
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{predicted_label}")
        if response.status_code == 200:
            data = response.json()
            animal_info = data.get("extract", "No additional information available.")
        else:
            animal_info = "No additional information available."
    except Exception:
        animal_info = "Failed to fetch additional information."
    st.markdown(f"**Info about {predicted_label}:** {animal_info}")

# Footer
st.markdown('<div class="footer">Made with ❤️ by Zach</div>', unsafe_allow_html=True)
