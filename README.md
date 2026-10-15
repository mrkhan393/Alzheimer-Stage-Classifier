# 🧠 Alzheimer Dementia Classifier (Streamlit)

An AI-powered web app that classifies brain MRI scans into dementia severity stages using a deep learning model, built with **Streamlit** and **TensorFlow**.

> ⚠️ **Disclaimer:** This project is a prototype built for educational purposes only. It is **not** a medical device and should not be used for real diagnosis or clinical decision-making. Always consult a qualified healthcare professional for medical advice.

## Overview

Upload a brain MRI image and the app will:

1. Validate that the uploaded image resembles an MRI scan (using perceptual image hashing against a set of reference MRIs).
2. Preprocess the image (grayscale, resize to 128×128).
3. Run it through a trained CNN (`model.keras`) to classify it into one of four categories.
4. Display the predicted class along with a probability breakdown for all classes.

## Visit
- **Live Site:** https://alzheimer-dimentia-classifier-app-dbbudbgwneodzvmhusvl8o.streamlit.app
- **GitHub Repository:** https://github.com/mrkhan393/Alzheimer-Stage-Classifier

## Classes

The model predicts one of the following four stages:

| Class | Label |
|-------|-------|
| 0 | Mild Demented |
| 1 | Moderate Demented |
| 2 | Non Demented |
| 3 | Very Mild Demented |

## Demo

The app UI includes:
- A drag-and-drop MRI image uploader
- A styled prediction card with a success message
- Animated probability bars for each class
- Basic validation to reject non-MRI images

## Tech Stack

- [Streamlit](https://streamlit.io/) – web app framework
- [TensorFlow / Keras](https://www.tensorflow.org/) – deep learning model inference
- [Pillow](https://python-pillow.org/) – image processing
- [ImageHash](https://github.com/JohannesBuchner/imagehash) – perceptual hashing for MRI validation
- [NumPy](https://numpy.org/) – array operations

## Project Structure

```
Alzheimer Dimentia Classifier Streamlit/
├── streamlit_app.py       # Main Streamlit application
├── model.keras            # Trained Keras classification model
├── reference_mri/         # Reference MRI images used for upload validation
├── requirements.txt       # Python dependencies
└── .gitignore
```

## How It Works

### 1. MRI Validation
Before running inference, the app checks whether the uploaded image "looks like" an MRI by comparing its average perceptual hash against a folder of reference MRI images (`reference_mri/`). If no reference image is within the hash distance threshold, the upload is rejected.

### 2. Preprocessing
The validated image is converted to grayscale, resized to `128x128`, and normalized to the `[0, 1]` range before being reshaped to match the model's expected input shape `(1, 128, 128, 1)`.

### 3. Prediction
The preprocessed image is passed to the trained Keras model, which outputs a probability distribution over the four dementia classes. The class with the highest probability is shown as the prediction, alongside a full breakdown of all class probabilities.

## Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/mrkhan393/Alzheimer-Dimentia-Classifier-Streamlit.git
cd Alzheimer-Dimentia-Classifier-Streamlit

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

### Usage

1. Launch the app.
2. Upload a brain MRI image (`.jpg`, `.jpeg`, or `.png`).
3. Click **🚀 Upload & Predict**.
4. View the predicted dementia stage and the probability breakdown.

## Model

The model (`model.keras`) is a CNN trained on grayscale MRI scans resized to 128×128 pixels, classifying images into 4 dementia stages. Training scripts/notebooks are not included in this repository — only the exported model artifact is provided for inference.

## Limitations

- The MRI validation step uses simple perceptual hashing, which is a heuristic and may occasionally reject valid MRIs or accept invalid ones.
- The model's predictions are based on a specific training dataset and may not generalize well to MRIs from different scanners, protocols, or populations.
- This tool has not been clinically validated and must not be used for actual diagnosis.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the [issues page](https://github.com/mrkhan393/Alzheimer-Dimentia-Classifier-Streamlit/issues) if you want to contribute.

## License

This project currently has no license specified.

## Acknowledgements

- Built with [Streamlit](https://streamlit.io/) and [TensorFlow](https://www.tensorflow.org/).
- Inspired by publicly available Alzheimer's MRI classification datasets (e.g., Kaggle's Alzheimer's Dataset).
