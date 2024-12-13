from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'gemstone_model.h5'
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 16

# Define class indices with actual class names
CLASS_INDICES = {
    0: 'Amethyst', 1: 'Aquamarine', 2: 'Citrine', 3: 'Diamond', 4: 'Emerald',
    5: 'Garnet', 6: 'Iolite', 7: 'Jade', 8: 'Lapis Lazuli', 9: 'Moonstone',
    10: 'Onyx', 11: 'Opal', 12: 'Peridot', 13: 'Ruby', 14: 'Sapphire',
    15: 'Spinel', 16: 'Tanzanite', 17: 'Topaz', 18: 'Tourmaline', 19: 'Zircon',
    20: 'Alexandrite', 21: 'Amazonite', 22: 'Andalusite', 23: 'Apatite', 24: 'Carnelian',
    25: 'Chrysoberyl', 26: 'Chrysoprase', 27: 'Danburite', 28: 'Fluorite', 29: 'Hematite',
    30: 'Kyanite', 31: 'Labradorite', 32: 'Malachite', 33: 'Morganite', 34: 'Obsidian',
    35: 'Pietersite', 36: 'Prehnite', 37: 'Quartz', 38: 'Rhodonite', 39: 'Sodalite',
    40: 'Spessartite', 41: 'Sugilite', 42: 'Sunstone', 43: 'Tigers Eye', 44: 'Turquoise',
    45: 'Variscite', 46: 'Zoisite', 47: 'Beryl', 48: 'Blue Topaz', 49: 'Brown Diamond',
    50: 'Canary Diamond', 51: 'Champagne Diamond', 52: 'Chocolate Diamond', 53: 'Color Change Garnet', 54: 'Coral',
    55: 'Fancy Sapphire', 56: 'Fire Opal', 57: 'Green Diamond', 58: 'Green Sapphire', 59: 'Kunzite',
    60: 'Lemon Quartz', 61: 'Mali Garnet', 62: 'Mandarin Garnet', 63: 'Paraiba Tourmaline', 64: 'Pink Diamond',
    65: 'Pink Sapphire', 66: 'Purple Sapphire', 67: 'Red Spinel', 68: 'Rose Quartz', 69: 'Tsavorite Garnet',
    70: 'Watermelon Tourmaline', 71: 'White Sapphire', 72: 'White Topaz', 73: 'Yellow Beryl', 74: 'Yellow Sapphire',
    75: 'Yellow Topaz', 76: 'Black Diamond', 77: 'Black Opal', 78: 'Cats Eye Chrysoberyl', 79: 'Color Change Sapphire',
    80: 'Color Change Spinel', 81: 'Fancy Diamond', 82: 'Fancy Garnet', 83: 'Gray Diamond', 84: 'Imperial Topaz',
    85: 'Orange Sapphire', 86: 'Peach Sapphire'
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded successfully.')

# Prediction function
def predict_gemstone(image_path):
    try:
        # Load the image and preprocess it
        img = image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_INDICES.get(predicted_class_index, "Unknown")
        return predicted_class
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e

# API endpoint to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Secure filename and save file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # Predict gemstone type from the uploaded image
        gemstone_type = predict_gemstone(file_path)
        return jsonify({'gemstoneType': gemstone_type, 'imagePath': f'/uploads/{filename}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve uploaded images
@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
