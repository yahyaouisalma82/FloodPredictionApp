from flask import Flask, request, jsonify
import pickle
import sklearn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes and origins

# Load the trained model
model = pickle.load(open('flood_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Welcome to the Flood Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the POST request
    data = request.json

    # Extract features from the data
    precipitation = data.get('precipitation')
    water_level = data.get('water_level')
    flow_rate = data.get('flow_rate')

    # Check if all necessary data is provided
    if precipitation is None or water_level is None or flow_rate is None:
        return jsonify({'error': 'Missing data'}), 400

    # Prepare the data for prediction
    features = [[precipitation, water_level, flow_rate]]

    # Make prediction using the model
    prediction = model.predict(features)[0]

    # Return the prediction result
    return jsonify({'prediction': 'Flood Likely' if prediction == 1 else 'No Flood'})

if __name__ == '__main__':
    app.run(debug=True)
