from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy ML model
X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])
model = LinearRegression()
model.fit(X, y)
print("Model trained. Coefficients:", model.coef_)
