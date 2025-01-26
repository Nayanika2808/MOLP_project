# Import the joblib library
import joblib

# Specify the path to the saved model file
model_file = 'models/best_model.joblib'  # Path to your saved model

# Try to load the model
try:
    # Load the model using joblib
    model = joblib.load(model_file)
    print("Model loaded successfully!")
    
    # Print the type of the loaded model to confirm it's correct
    print(f"Model type: {type(model)}")

except Exception as e:
    print(f"An error occurred while loading the model: {e}")