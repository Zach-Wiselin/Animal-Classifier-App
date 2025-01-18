# Animal Classifier App 🐾

Welcome to the Animal Classifier App! This application uses machine learning and computer vision to identify animals in real-time from your webcam feed and provides fun, informative facts about them. Powered by Streamlit, YOLOv5, and a subset of COCO dataset classes, this app is a fantastic tool for animal enthusiasts and learners.

---

## Features

- **Real-time Animal Detection**: Capture live webcam feed and classify animals with high accuracy.
- **Animal Facts**: Learn fun and interesting facts about the detected animals.
- **Interactive Interface**: Built with Streamlit for a clean and user-friendly experience.

---

## How to Set Up and Run the App

Follow these steps to get the app up and running on your local machine:

### **Step 1: Clone the Repository**

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Zach-Wiselin/Animal-Classifier-App.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Animal-Classifier-App
   ```

### **Step 2: Clone the YOLOv5 Repository**

1. Clone the YOLOv5 repository into the project folder:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   ```
2. Navigate to the YOLOv5 directory and install its dependencies:
   ```bash
   cd yolov5
   pip install -r requirements.txt
   ```
3. Return to the main project directory:
   ```bash
   cd ..
   ```

### **Step 3: Install Dependencies**

1. Install the Python dependencies specified in the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

### **Step 4: Run the Application**

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. A new browser window/tab will open, displaying the app interface.

---

## Application Workflow

1. **Start Webcam Feed**: Check the "Start Webcam" box to activate your webcam and begin detecting animals.
2. **Real-time Detection**: The app will classify animals in the webcam feed and display their names and confidence scores.
3. **Learn About Animals**: When an animal is detected, the app fetches fun facts and information about it using the Wikipedia API.
4. **Retry Detection**: Use the "Retry" button to clear the current detection and start over.

---

## Project Structure

Here's an overview of the project's structure:

```
Animal-Classifier-App/
├── app.py              # Main application script
├── requirements.txt    # List of dependencies
├── README.md           # Documentation (this file)
├── yolov5/             # Cloned YOLOv5 repository
```

---

## Troubleshooting

### **Common Issues**

1. **YOLOv5 Repository Not Found**:
   Ensure that the YOLOv5 repository is cloned inside the project directory. Follow Step 2 above.

2. **Webcam Access Denied**:
   Ensure your browser or system allows webcam access. If using a remote server, additional configurations may be required.

3. **ModuleNotFoundError**:
   Double-check that all dependencies are installed with:
   ```bash
   pip install -r requirements.txt
   ```

4. **Wikipedia API Issues**:
   Ensure an internet connection is active to fetch animal information.

---

## Future Improvements

- Support for additional animal classes.
- Integration with more datasets for enhanced accuracy.
- Improved UI for better user experience.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Made with ❤️ by Zach. Happy Exploring!
