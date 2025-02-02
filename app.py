from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model and encoder
model = joblib.load('flood_prediction_model.pkl')  # Load the trained model
encoder = joblib.load('encoder (1).pkl')  # Load the encoder

@app.route('/')
def home():
    return render_template("index.html")

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from frontend
        data = request.get_json(force=True)
        annual_rainfall = np.array(data['features']).reshape(1, -1)  # Convert to 2D array for prediction
        
        # Here we assume you have other categorical features in the dataset. 
        # We handle encoding the features based on the encoder used during training.
        
        # Encode categorical features (if applicable)
        # Assuming all categorical features are from the training data
        encoded_features = encoder.transform(annual_rainfall)

        # Make the prediction using the trained model
        prediction = model.predict(encoded_features)

        # Return result as JSON
        if prediction[0] == 1:
            flood_status = "Flood will occur"
        else:
            flood_status = "Flood will not occur"
        
        return jsonify({'prediction': flood_status})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
