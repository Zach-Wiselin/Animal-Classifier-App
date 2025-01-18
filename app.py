import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Load the iNaturalist species classifier model from TensorFlow Hub
@st.cache_resource
def load_inaturalist_model():
    return tf.keras.models.load_model("https://tfhub.dev/google/inaturalist_species_classification/1")

model = load_inaturalist_model()

# Preprocess the image for the iNaturalist model
def preprocess_image(image):
    image = image.resize((299, 299))  # iNaturalist model expects 299x299 input
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Class labels for the iNaturalist model (manually added for this example)
animal_labels = ["cow", "elephant", "zebra", "giraffe", "lion", "tiger", "deer", "fox", "rabbit", "wolf", "bear", "monkey", "panda", "kangaroo", "sheep", "peacock", "parrot", "eagle"]

# Streamlit UI
st.title("Animal Classifier 🐾")
st.write("Capture an image using your webcam to classify animals accurately.")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Load and preprocess the captured image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

    input_array = preprocess_image(image)

    # Predict using the iNaturalist model
    predictions = model.predict(input_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = animal_labels[predicted_class_idx] if predicted_class_idx < len(animal_labels) else "Unknown"
    confidence = predictions[0][predicted_class_idx] * 100

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
