import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the model
try:
    with open("best_model.pkl", "rb") as f:  # Ensure the correct file extension
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file not found.")
    exit(1)
except Exception as e:
    print("Error loading the model:", e)
    exit(1)

# Example test data (ensure this matches the model's expected input shape)
X_test = np.array([[1.2, 3.4, 5.6, 7.8], [2.3, 4.5, 6.7, 8.9]])  # Test input features
y_test = np.array([10, 20])  # Test target values

# Check input consistency
if X_test.shape[1] != model.coef_.shape[0]:
    print(f"Error: Model expects {model.coef_.shape[0]} features, but got {X_test.shape[1]}.")
    exit(1)

# Make predictions
try:
    predictions = model.predict(X_test)
    print("Predictions:", predictions)
except Exception as e:
    print("Error during prediction:", e)
    exit(1)

# Evaluate performance
try:
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)
except Exception as e:
    print("Error during evaluation:", e)
    exit(1)