import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests

# Load MobileNetV2 pre-trained on ImageNet
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# Load ImageNet class labels
@st.cache_data
def load_labels():
    labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    response = requests.get(labels_url)
    response.raise_for_status()
    labels = response.text.splitlines()
    return labels

labels = load_labels()

# Define animal-related ImageNet labels (manually filtered)
animal_classes = [
    "golden retriever", "tabby cat", "Persian cat", "cow", "elephant", "zebra", "giraffe", "tiger",
    "lion", "cheetah", "bear", "panda", "kangaroo", "wolf", "fox", "deer", "rabbit", "monkey",
    "peacock", "parrot", "eagle", "owl", "penguin", "sparrow", "flamingo", "hawk", "woodpecker"
]

# Preprocess image for MobileNetV2
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to MobileNetV2 input size
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit App
st.title("Animal Classifier 🐾")
st.write("Capture an image using your webcam to classify animals.")

# Webcam input
camera_input = st.camera_input("Take a picture using your webcam")

if camera_input:
    # Load and preprocess image
    image = Image.open(camera_input).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)
    input_array = preprocess_image(image)

    # Predict using MobileNetV2
    predictions = model.predict(input_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # Check if the predicted label is an animal
    if predicted_label in animal_classes:
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
    else:
        st.markdown("### Sorry, this doesn't seem to be an animal.")
        st.markdown(f"Detected: {predicted_label} (Confidence: {confidence:.2f}%)")

# Footer
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ by Zach</div>", unsafe_allow_html=True)
