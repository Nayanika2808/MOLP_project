import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load preprocessed data
input_file = "data/preprocessed_data.csv"
output_file = "model.pkl"

try:
    # Read the dataset
    data = pd.read_csv(input_file)
    print(f"Dataset loaded successfully from {input_file}")
    
    # Verify columns
    expected_columns = ['feature', 'target']
    if not all(col in data.columns for col in expected_columns):
        raise KeyError(f"Expected columns {expected_columns} not found in dataset. Found columns: {list(data.columns)}")

    # Extract features and target
    X = data[['feature']].values  # Replace 'feature' with the actual column name if needed
    y = data['target'].values    # Replace 'target' with the actual column name if needed

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    print("Model training complete.")

    # Save the trained model
    with open(output_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved successfully to {output_file}")

except FileNotFoundError:
    print(f"Error: Input file '{input_file}' not found. Please ensure the file exists.")
except KeyError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")