import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load MobileNetV2 model pre-trained on ImageNet
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# Load ImageNet labels
@st.cache_data
def load_labels():
    # These are the ImageNet class labels
    labels_path = tf.keras.utils.get_file(
        "ImageNetLabels.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    )
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# Preprocess image for MobileNetV2
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to MobileNetV2 input size
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocess for MobileNetV2
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit App
st.title("Animal Classifier 🐾")
st.write("Capture an image using your webcam to classify animals.")

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
    predicted_label = labels[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100

    # Display results
    st.write(f"### Predicted: {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Provide additional context for animals
    if predicted_label.lower() in [label.lower() for label in labels if "dog" in label or "cat" in label or "bird" in label]:
        st.write(f"Fetching additional information about {predicted_label}...")
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
st.markdown("<div style='text-align: center; margin-top: 50px;'>Made with ❤️ using TensorFlow MobileNetV2</div>", unsafe_allow_html=True)
