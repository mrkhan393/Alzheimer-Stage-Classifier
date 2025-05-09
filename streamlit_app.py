import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import imagehash

# Load model
model = tf.keras.models.load_model('model.keras')

# Class labels
CLASS_NAMES = {
    0: 'Mild Demented',
    1: 'Moderate Demented',
    2: 'Non Demented',
    3: 'Very Mild Demented'
}

# Function to validate MRI image
def is_mri_like_uploaded_image(image, reference_folder="reference_mri", threshold=10):
    try:
        uploaded_hash = imagehash.average_hash(image.convert('L'))
        for ref_file in os.listdir(reference_folder):
            if ref_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                ref_path = os.path.join(reference_folder, ref_file)
                ref_image = Image.open(ref_path).convert('L')
                ref_hash = imagehash.average_hash(ref_image)
                if abs(uploaded_hash - ref_hash) < threshold:
                    return True
        return False
    except Exception as e:
        st.error(f"Validation Error: {e}")
        return False

# Prediction function
def classify_image(image):
    img = image.convert('L').resize((128, 128))
    img_array = img_to_array(img) / 255.0

    if not is_mri_like_uploaded_image(image):
        return None, None, "Uploaded image does not appear to be a valid MRI scan."

    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 128, 128, 1)
    predictions = model.predict(img_array)[0]
    predicted_class = int(np.argmax(predictions))
    return predicted_class, predictions, None

# --- Streamlit Custom Styling ---
st.set_page_config(page_title="🧠 MRI Dementia Classifier", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #3b5e94;
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background: rgba(0, 0, 0, 0.5);
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            color: white;
        }
        h2, h5 {
            color: white;
        }
        .uploaded-img {
            border-radius: 10px;
            border: 2px solid #fff;
            margin-top: 10px;
            width: 200px;
            height: 190px;
            object-fit: cover;
        }
        .bar {
            background-color: #f1f1f1;
            border-radius: 8px;
            height: 20px;
            overflow: hidden;
        }
        .bar-fill {
            background-color: #2E86AB;
            height: 100%;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main">
        <h2 style='text-align: center;'>🧠 AI-Powered MRI Dementia Classifier</h2>
        <p style='text-align: center;'>Upload an MRI scan to classify possible dementia type using a trained deep learning model.</p>
    </div>
""", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=200, caption="Uploaded MRI (200×190)", output_format="JPEG")

    if st.button("🚀 Upload & Predict"):
        with st.spinner("Analyzing image..."):
            pred_class, probs, error = classify_image(image)
            if error:
                st.error(error)
            else:
                st.success(f"Prediction: {CLASS_NAMES[pred_class]}")

                st.markdown("### Class Probabilities")
                for i, prob in enumerate(probs):
                    st.markdown(f"""
                        <div style='margin-bottom: 10px;'>
                            <b>{CLASS_NAMES[i]}</b>
                            <div class='bar'>
                                <div class='bar-fill' style='width: {prob * 100:.1f}%;'></div>
                            </div>
                            <small>{prob * 100:.2f}%</small>
                        </div>
                    """, unsafe_allow_html=True)
else:
    st.info("Please upload a valid MRI image to begin classification.")

# Footer
st.markdown("""
    <hr>
    <div class='footer'>
        Built with Streamlit & TensorFlow | A prototype for educational purposes
    </div>
""", unsafe_allow_html=True)
