from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and scaler (ensure these files are in your local directory)
model = joblib.load('flood_prediction_model.pkl')  
scaler = joblib.load('scaler.pkl')  # Load scaler if used

@app.route('/')
def home():
    return "Flood Prediction API is running locally!"

# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)  # Convert to array

        # Apply scaling if necessary
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)

        # Return result as JSON
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
