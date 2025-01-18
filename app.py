import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import os

# GitHub URL for the model file
MODEL_URL = "https://raw.githubusercontent.com/Zach-Wiselin/Animal-Classifier-App/main/Model.h5"

# Download the model if not present locally
@st.cache_resource
def load_model():
    if not os.path.exists("Model.h5"):
        with st.spinner("Downloading model..."):
            try:
                response = requests.get(MODEL_URL)
                with open("Model.h5", "wb") as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Failed to download the model: {e}")
                st.stop()

    # Load the model
    try:
        return tf.keras.models.load_model("Model.h5")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# Preprocess the image for the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Ensure image size matches model input
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit App
st.title("Animal Classifier 🐾")
st.write("Capture an image using your webcam to classify animals.")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Load and preprocess the captured image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    input_array = preprocess_image(image)

    # Predict using the model
    predictions = model.predict(input_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100

    # Update with the actual class labels
    animal_labels = ["lion", "tiger", "cow", "elephant", "zebra", "giraffe", "bear", "deer", "fox", "rabbit"]

    # Map the prediction index to the label
    predicted_label = animal_labels[predicted_class_idx] if predicted_class_idx < len(animal_labels) else "Unknown"

    # Display the prediction
    st.markdown(f"### It's a **{predicted_label.capitalize()}**!")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Fetch animal information from Wikipedia
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
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ by Zach</div>", unsafe_allow_html=True)
