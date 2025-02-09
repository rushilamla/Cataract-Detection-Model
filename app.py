from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib
from PIL import Image
import io

# Load the trained Logistic Regression model and scaler
model = joblib.load("cataract_model.pkl")
scaler = joblib.load("scaler.pkl")

# Flask app setup
app = Flask(__name__)

def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((128, 128))  # Resize to match training data
    image = np.array(image).flatten()  # Flatten the image
    image = image / 255.0  # Normalize pixel values
    image = scaler.transform([image])  # Standardize using trained scaler
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file'].read()
    image = preprocess_image(file)
    
    prediction = model.predict(image)
    result = "Cataract Detected" if prediction[0] == 1 else "Healthy Eye"
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)