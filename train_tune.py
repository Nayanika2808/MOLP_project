import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import pickle

# Define file paths
DATA_FILE = "data/preprocessed_data.csv"
MODEL_FILE = "best_model.pkl"

try:
    # Load preprocessed data
    data = pd.read_csv(DATA_FILE)
    print(f"Data loaded successfully from {DATA_FILE}")
    
    # Check dataset size
    if len(data) < 2:
        raise ValueError("Dataset must have at least 2 samples for R² score to be well-defined.")
    
    # Ensure required columns are present
    required_columns = ['feature', 'target']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Extract features and target
    X = data[['feature']].values
    y = data['target'].values
    
    # Define the model and hyperparameters
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}
    
    # Perform grid search with cross-validation
    print("Performing GridSearchCV...")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    
    # Best parameters and model
    print("Grid search complete.")
    print("Best Parameters:", grid.best_params_)
    best_model = grid.best_estimator_
    
    # Save the best model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved as {MODEL_FILE}")

except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
except ValueError as ve:
    print(f"Value Error: {ve}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
