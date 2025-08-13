"""
Linear Regression Implementation for Diabetes Dataset
====================================================
This script demonstrates multivariate linear regression using scikit-learn
to predict diabetes progression based on patient physiological features.
Includes comprehensive visualization and feature importance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# Load diabetes dataset (10 features: age, sex, BMI, etc.)
diabetes = datasets.load_diabetes()
print(f"Dataset shape: {diabetes.data.shape}")
print(f"Features: {diabetes.feature_names}")

# Split data (70/30) with randomization
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, 
    test_size=0.3, random_state=42
)

# Train model and evaluate
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Results
r2 = model.score(X_test, y_test)
mse = mean_squared_error(y_test, predictions)
print(f"R² Score: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {np.sqrt(mse):.3f}")

# Visualization
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, predictions, alpha=0.7, color='blue', edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
residuals = y_test - predictions
plt.scatter(predictions, residuals, alpha=0.7, color='green', edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': diabetes.feature_names,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance)