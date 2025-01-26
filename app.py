from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
MODEL_FILE = "best_model.pkl"

try:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found. Please ensure the file exists.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting a value based on input features."""
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        if not data or 'feature' not in data:
            return jsonify({"error": "Invalid input. JSON body must contain a 'feature' key."}), 400

        # Extract feature and convert to a 2D array for the model
        feature = data['feature']
        if not isinstance(feature, (int, float, list)):
            return jsonify({"error": "'feature' must be a number or a list of numbers."}), 400

        # Handle single feature vs multiple features
        if isinstance(feature, (int, float)):
            feature = [[feature]]
        else:
            feature = np.array(feature).reshape(-1, 1)

        # Predict using the model
        prediction = model.predict(feature).tolist()

        # Return prediction as JSON
        return jsonify({"predictions": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if model is not None:
        return jsonify({"status": "Healthy"}), 200
    return jsonify({"status": "Model not loaded"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)