# Animal Classifier App 🐾

Welcome to the **Animal Classifier App**! This application uses machine learning and computer vision to classify animals in real-time using a webcam feed. It also provides interesting facts about the identified animals by integrating with the Wikipedia API. Built with Streamlit, this app offers a clean and interactive user interface for animal enthusiasts and learners.

Check out the live app here: **Animal Classifier App**

---

## Features

- **Real-time Animal Classification:** Capture an image using your webcam, and the app identifies the animal in the image.
- **Informative Facts:** Fetches additional information about the detected animal from Wikipedia.
- **Elegant Design:** Clean and responsive interface with centralized elements for an enhanced user experience.

---

## How the Project Works

### Input Image Capture
- The app uses Streamlit's `st.camera_input()` feature to allow users to capture an image using their webcam.

### Image Preprocessing
- Captured images are resized to **224x224 pixels**, normalized, and converted into a format compatible with the pre-trained model.

### Animal Classification
- The app uses TensorFlow Hub's **MobileNetV2** model to classify animals in the captured image. This lightweight, pre-trained model is hosted on TensorFlow Hub and designed for image classification tasks.

### Wikipedia Integration
- After predicting the animal, the app uses the **Wikipedia API** to fetch additional information about the identified animal and displays it to the user.

### Interactive Feedback
- The app displays the predicted animal's name, confidence score, and relevant information in a user-friendly layout.

---

## Tools and Technologies Used

1. **Streamlit**: Provides a clean and interactive user interface for capturing webcam images and displaying classification results.
2. **TensorFlow Hub**: The app leverages the MobileNetV2 model hosted on TensorFlow Hub for efficient and accurate image classification.
3. **Pillow (PIL)**: Used for image manipulation and preprocessing.
4. **NumPy**: Handles numerical operations required for image normalization and reshaping.
5. **Wikipedia API**: Fetches detailed information about the identified animal for enhanced user engagement.

---

## How to Set Up and Run the App

### Step 1: Clone the Repository
Clone this repository to your local machine:

```bash
git clone https://github.com/Zach-Wiselin/Animal-Classifier-App.git
```

Navigate to the project directory:

```bash
cd Animal-Classifier-App
```

### Step 2: Install Dependencies
Install the required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Start the Streamlit app:

```bash
streamlit run app.py
```

A new browser window/tab will open, displaying the app interface.

---

## Application Workflow

1. **Start the App:** Open the app in your browser after starting it locally.
2. **Capture an Image:** Use the "Take a Picture" button to capture an image of an animal using your webcam.
3. **View Classification Results:** The app will classify the captured image and display:
   - The predicted animal's name.
   - The confidence score of the prediction.
   - Fun and informative details about the animal fetched from Wikipedia.
4. **Retry:** Capture another image for classification.

---

## Troubleshooting

### Common Issues

- **Webcam Not Working:**
  - Ensure your browser or system allows webcam access.

- **ModuleNotFoundError:**
  - Double-check that all dependencies are installed using:

    ```bash
    pip install -r requirements.txt
    ```

- **Wikipedia API Issues:**
  - Ensure your internet connection is active.

---

## Future Improvements

- Add support for a broader range of animal classes.
- Incorporate multiple pre-trained models for enhanced classification accuracy.
- Improve UI responsiveness and animations for better user interaction.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Made with ❤️ by Zach**

