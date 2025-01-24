from sklearn.linear_model import LinearRegression
import numpy as np
import mlflow
import mlflow.sklearn

# Dummy ML model
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])

# Start an MLflow experiment
mlflow.set_experiment("Simple Experiment")

with mlflow.start_run():
    # Define and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Log parameters and metrics
    mlflow.log_param("fit_intercept", model.fit_intercept)
    mlflow.log_metric("score", model.score(X, y))
    mlflow.log_metric("coefficient", model.coef_[0])

    # Save the model
    mlflow.sklearn.log_model(model, "linear_model")

# Print model coefficients after logging is complete
print("Model trained. Coefficients:", model.coef_)
