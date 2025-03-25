import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask("LUNGCANCERAI")
CORS(app)

# Load the trained model from the models folder
model_path = os.path.join("models", "lung_cancer_improvedtrain_cnn_complete.h5")
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    """
    Preprocess the input image to match the model's expected input.
    - Converts to RGB
    - Resizes to 224x224
    - Normalizes pixel values to [0, 1]
    - Adds a batch dimension
    """
    img = image.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint to verify that the API is running and the model is loaded.
    """
    return jsonify({"model_loaded": True, "status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint that accepts an image file via a form-data request.
    The image is preprocessed and passed to the model for prediction.
    Returns the predicted class and the confidence scores.
    """
    try:
        # Check if an image file is included in the request
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400

        file = request.files["file"]

        # Read the image file and preprocess it
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        # Run the prediction
        prediction = model.predict(processed_image)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        # Return the prediction result and confidence scores
        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "confidence": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/save_scan", methods=["POST"])
def save_scan():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
        
    file = request.files["file"]
    patient_id = request.form.get("patient_id")  # This will be your Firebase patient ID
    
    # Create directory if it doesn't exist
    patient_dir = os.path.join("patient_data", patient_id)
    os.makedirs(patient_dir, exist_ok=True)
    
    # Save file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    file_path = os.path.join(patient_dir, filename)
    file.save(file_path)
    
    # Process image and get prediction
    image = Image.open(file_path)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    
    # Return the prediction and file path
    return jsonify({
        "status": "success",
        "prediction": predicted_class,
        "confidence": prediction.tolist(),
        "file_path": file_path
    })


if __name__ == "__main__":
    app.run(debug=True, port=3000)
