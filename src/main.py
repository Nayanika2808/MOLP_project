from sklearn.linear_model import LinearRegression
import numpy as np
import mlflow
import mlflow.sklearn

# Dummy data for training
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# Set up MLflow experiment
mlflow.set_experiment("Multiple Runs Experiment")

# Different configurations for model training
configurations = [
    {"fit_intercept": True},
    {"fit_intercept": False},
    {"fit_intercept": True, "additional_data": True},
]

# Additional data for one configuration
X_additional = np.array([[6], [7], [8]])
y_additional = np.array([6, 7, 8])

# Iterate through configurations
for i, config in enumerate(configurations):
    with mlflow.start_run(run_name=f"Run {i + 1}"):
        # Prepare data (optional: add extra data for some runs)
        if config.get("additional_data", False):
            X_train = np.vstack((X, X_additional))
            y_train = np.concatenate((y, y_additional))
        else:
            X_train = X
            y_train = y

        # Train model
        model = LinearRegression(fit_intercept=config["fit_intercept"])
        model.fit(X_train, y_train)

        # Log parameters
        mlflow.log_param("fit_intercept", config["fit_intercept"])
        if config.get("additional_data", False):
            mlflow.log_param("additional_data", True)

        # Log metric
        mlflow.log_metric("score", model.score(X_train, y_train))
        mlflow.log_metric("coefficient", model.coef_[0])

        # Log the model
        mlflow.sklearn.log_model(model, f"model_run_{i + 1}")

        print(f"Run {i + 1} complete. Model Coefficient: {model.coef_[0]}")
