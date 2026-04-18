# 🤟 ASL Hand Sign Recognition System

![Python](https://img.shields.io/badge/python-v3.12-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-green.svg)

A real-time American Sign Language (ASL) detection application built with **Streamlit**, **MediaPipe Hand Landmarks**, and **Scikit-Learn (RandomForest)**. The application captures your webcam feed in the browser, extracts your hand skeletal landmarks, and classifies the current ASL alphabet sign.

## ✨ Features
* **Real-time Webcam Inference**: Uses OpenCV and MediaPipe to track hands live.
* **Full Alphabet Support**: Supports A-Z signs, as well as `space`, `del`, and `nothing`.
* **Sentence Builder**: Hold a sign steady for ~1 second to automatically append it to a running sentence. Includes integrated spacing and letter deletion controls.
* **Modern UI**: Features a dark-glassmorphism Streamlit UI with confidence bars, EMA (Exponential Moving Average) prediction smoothing, custom camera flip toggles, and live FPS tracking.
* **Visual Guides**: Built-in visual sidebar with ASL reference sheets for learning alongside the model.

## 🚀 Getting Started

### 1. Requirements

Ensure you have Python installed. It is highly recommended to use a virtual environment.

```bash
# Optional: Setup virtual environment
python -m venv asl_venv
asl_venv\Scripts\activate  # Windows
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies generally include `streamlit`, `opencv-python`, `mediapipe`, `scikit-learn`, `numpy`, and `joblib`)*

### 2. Large File Requirements (Models)
**Note:** The trained machine learning classification models (e.g. `landmark_classifier.pkl`, `landmark_classifier2.pkl`) and extracted dataset CSVs exceed GitHub's absolute 100MB file limit. 

To run the application locally, you must either:
1. Obtain the `.pkl` model file and place it in the `asl_venv/` directory as `landmark_classifier2.pkl`.
2. Generate a new model locally by running the extraction and training scripts (`data_extractor.py` followed by `train_landmark_model.py` / `train.py`).

### 3. Run the Interface
To launch the Web Application, simply run:
```bash
streamlit run app.py
```
This will open the application in your default local web browser. Simply click **"▶ Start Detection"** to activate the webcam!

## 🗂️ Project Structure
* `app.py`: The main Streamlit web application.
* `model.py` / `train.py`: Primary machine learning classification logic and training pipes.
* `extract.py` / `data_extractor.py`: Utility scripts used to turn image datasets into MediaPipe numerical landmarks and save them to CSV.
* `live_detect.py` / `test.py`: Standalone opencv-based testing scripts outside of the Streamlit ecosystem.
* `requirements.txt`: Package dependency mappings.

## 🤝 Contributing
Contributions are welcome! If you'd like to help increase the accuracy or add additional signs (e.g., dynamic signs or words), feel free to fork the repository and open a pull request.
