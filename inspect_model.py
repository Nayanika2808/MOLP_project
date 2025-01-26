import os
import joblib

file_path = 'models/best_model.joblib'  # Path to model file in 'models/' folder

if os.path.exists(file_path):
    print(f"File exists at {file_path}")
    # Load the model
    model = joblib.load(file_path)
    print("Model loaded successfully!")
    # Inspect the model
    print(f"Model type: {type(model)}")
    if hasattr(model, 'get_params'):  # Check if model has get_params method
        print(f"Model parameters: {model.get_params()}")
    else:
        print("Model does not have parameters to show.")
else:
    print(f"File does NOT exist at {file_path}")