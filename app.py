from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import imagehash

app = Flask(__name__)
model = tf.keras.models.load_model('model.keras')

# Class label mapping
CLASS_NAMES = {
    0: 'MildDemented',
    1: 'ModerateDemented',
    2: 'NonDemented',
    3: 'VeryMildDemented'
}

# Optional: Verify image resembles an MRI
def is_mri_like_uploaded_image(upload_path, reference_folder="reference_mri", threshold=10):
    try:
        uploaded_hash = imagehash.average_hash(Image.open(upload_path).convert('L'))
        for ref_file in os.listdir(reference_folder):
            if ref_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                ref_path = os.path.join(reference_folder, ref_file)
                ref_hash = imagehash.average_hash(Image.open(ref_path).convert('L'))
                if abs(uploaded_hash - ref_hash) < threshold:
                    return True
        return False
    except Exception as e:
        print(f"Image validation error: {e}")
        return False

def classify_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((128, 128))
    image = img_to_array(image) / 255.0

    if not is_mri_like_uploaded_image(image_path):
        return None, None, "Uploaded image doesn't appear to be an MRI scan."

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = int(np.argmax(prediction[0]))
    return predicted_class, prediction[0], None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        predicted_class, probabilities, error = classify_image(file_path)
        if error:
            return jsonify({'error': error}), 400

        result = {
            'predicted_class': CLASS_NAMES[predicted_class],
            'probabilities': {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
