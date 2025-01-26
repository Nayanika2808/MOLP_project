import numpy as np
import joblib

# Step 1: Prepare the Test Data (update with correct number of features)
# Example: Age, BMI, Gender (0: Male, 1: Female), Salary
test_data = np.array([
    [45, 30, 1, 55000],  # Test sample 1
    [23, 22, 0, 32000],  # Test sample 2
    [60, 28, 1, 75000]   # Test sample 3
])

# Step 2: Load the Pre-trained Model
model = joblib.load('models/best_model.joblib')

# Validate the number of features the model expects
print(f"Model expects {model.n_features_in_} features.")

# Step 3: Make Predictions
predictions = model.predict(test_data)

# Step 4: Print the Predictions
print("Predictions:", predictions)