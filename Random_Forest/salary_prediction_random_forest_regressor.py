"""
Random Forest Regression for Salary Prediction
==============================================
This script demonstrates regression using Random Forest to predict salary
based on position level. Includes comprehensive evaluation, feature importance
analysis, and proper train/test methodology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load and explore the dataset
data = pd.read_csv('SalaryForRF.csv')
print("Salary Dataset Overview:")
print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nFirst 10 rows:")
print(data.head(10))

# Check for missing values and basic statistics
print(f"\nMissing values: {data.isnull().sum().sum()}")
print(f"\nDataset statistics:")
print(data.describe())

# Prepare features and target with proper column names
X = data.iloc[:, 1:2].values  # Position level
y = data.iloc[:, 2].values    # Salary

# Convert to DataFrame for better handling
feature_names = ['Position_Level']
X_df = pd.DataFrame(X, columns=feature_names)

print(f"\nFeature range: {X.min():.1f} - {X.max():.1f}")
print(f"Salary range: ${y.min():,.0f} - ${y.max():,.0f}")

# Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

print(f"\nData Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create and train Random Forest Regressor
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    oob_score=True
)

rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred_train = rf_regressor.predict(X_train)
y_pred_test = rf_regressor.predict(X_test)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training MSE: ${train_mse:,.0f}")
print(f"Testing MSE: ${test_mse:,.0f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")
print(f"Testing MAE: ${test_mae:,.0f}")

# Check for overfitting
r2_difference = train_r2 - test_r2
if r2_difference > 0.2:
    print("⚠️ Model may be overfitting!")
else:
    print("✅ Model generalizes well!")

# Cross-validation for robust evaluation
cv_scores = cross_val_score(rf_regressor, X_df, y, cv=5, scoring='r2')
print(f"\nCross-Validation Results:")
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance (even though only one feature)
print(f"\nFeature Importance:")
print(f"Position Level: {rf_regressor.feature_importances_[0]:.3f}")
print(f"Out-of-bag score: {rf_regressor.oob_score_:.3f}")

# Prediction function
def predict_salary(position_level):
    """Predict salary based on position level"""
    input_data = pd.DataFrame({'Position_Level': [position_level]})
    prediction = rf_regressor.predict(input_data)
    
    print(f"\nPrediction for Position Level {position_level}:")
    print(f"Estimated Salary: ${prediction[0]:,.0f}")
    return prediction[0]

# Create smooth curve for visualization (fix the range issue)
X_min, X_max = float(X.min()), float(X.max())
X_grid = np.arange(X_min, X_max + 0.1, 0.1).reshape(-1, 1)
X_grid_df = pd.DataFrame(X_grid, columns=feature_names)
y_grid_pred = rf_regressor.predict(X_grid_df)

# Comprehensive visualization
plt.figure(figsize=(15, 10))

# Main prediction plot
plt.subplot(2, 3, 1)
plt.scatter(X_train, y_train, color='red', s=100, alpha=0.8, label='Training Data', edgecolors='black')
plt.scatter(X_test, y_test, color='orange', s=100, alpha=0.8, label='Test Data', edgecolors='black')
plt.plot(X_grid, y_grid_pred, color='blue', linewidth=2, label='Random Forest Prediction')
plt.title('Salary vs Position Level\n(Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Training vs Test Performance (R²)
plt.subplot(2, 3, 2)
performance_data = ['Training', 'Testing']
r2_scores = [train_r2, test_r2]
bars = plt.bar(performance_data, r2_scores, color=['lightgreen', 'lightcoral'], edgecolor='black')
plt.title('R² Score: Training vs Testing')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2_scores[i]:.3f}', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

# MSE Comparison
plt.subplot(2, 3, 3)
mse_scores = [train_mse, test_mse]
bars = plt.bar(performance_data, mse_scores, color=['skyblue', 'lightpink'], edgecolor='black')
plt.title('MSE: Training vs Testing')
plt.ylabel('Mean Squared Error ($)')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_scores)*0.02, 
             f'${mse_scores[i]:,.0f}', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

# Cross-validation scores
plt.subplot(2, 3, 4)
plt.bar(range(len(cv_scores)), cv_scores, color='gold', edgecolor='black')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation R² Scores')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(2, 3, 5)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.7, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Predicted Salary ($)')
plt.ylabel('Residuals ($)')
plt.grid(True, alpha=0.3)

# Actual vs Predicted
plt.subplot(2, 3, 6)
plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted Salary')
plt.xlabel('Actual Salary ($)')
plt.ylabel('Predicted Salary ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example predictions for different position levels
print(f"\nExample Predictions:")
test_positions = [2.5, 4.0, 6.5, 8.0, 9.5]
for position in test_positions:
    predict_salary(position)

# Model information
print(f"\nModel Information:")
print(f"Number of estimators: {rf_regressor.n_estimators}")
print(f"Max depth: {rf_regressor.max_depth}")
print(f"Min samples split: {rf_regressor.min_samples_split}")
print(f"Min samples leaf: {rf_regressor.min_samples_leaf}")

# Performance summary
print(f"\nPerformance Summary:")
print(f"Model explains {test_r2*100:.1f}% of salary variance")
print(f"Average prediction error: ${test_mae:,.0f}")
print(f"RMSE: ${np.sqrt(test_mse):,.0f}")