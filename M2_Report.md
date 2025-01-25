## Title 
Experiment Tracking with MLflow

## Introduction
The purpose of this experiment was to track parameters, metrics, and models for three different configurations of a Linear Regression model using MLflow. The configurations varied based on the inclusion of an intercept term (`fit_intercept`) and additional training data.

## Experiment Setup

- **Experiment Name:** Multiple Runs Experiment
- **Tool Used:** MLflow
- **Model:** Linear Regression
- **Configurations:**
  - **Run 1:** `fit_intercept=True`, no additional data
  - **Run 2:** `fit_intercept=False`, no additional data
  - **Run 3:** `fit_intercept=True`, with additional data

## Results

| Run Name | fit_intercept | additional_data | score  | coefficient |
|----------|---------------|-----------------|--------|-------------|
| Run 1    | True          | False           | 1.0    | 1.0         |
| Run 2    | False         | False           | 0.9    | 0.9         |
| Run 3    | True          | True            | 1.0    | 1.0         |

### Observations:
1. **Run 1:** The default configuration performed perfectly on the original dataset.
2. **Run 2:** Disabling the intercept term resulted in a slight drop in performance.
3. **Run 3:** Adding additional training data maintained a perfect score, demonstrating robustness.
   
![image](https://github.com/user-attachments/assets/b3bb2167-5958-4ca7-a6fa-317cff9b95fa)
