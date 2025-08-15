"""
Decision Tree Regression for Game Development Profit Prediction
==============================================================
This script demonstrates decision tree regression using scikit-learn
to predict game development profits based on production costs.
Includes proper train/test evaluation and overfitting prevention.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Game development data: [Game Type, Production Cost, Expected Profit]
game_data = np.array([
    ['Asset Flip', 100, 1000],
    ['Text Based', 500, 3000],
    ['Visual Novel', 1500, 5000],
    ['2D Pixel Art', 3500, 8000],
    ['2D Vector Art', 5000, 6500],
    ['Strategy', 6000, 7000],
    ['First Person Shooter', 8000, 15000],
    ['Simulator', 9500, 20000],
    ['Racing', 12000, 21000],
    ['RPG', 14000, 25000],
    ['Sandbox', 15500, 27000],
    ['Open-World', 16500, 30000],
    ['MMOFPS', 25000, 52000],
    ['MMORPG', 30000, 80000]
])

print("Game Development Dataset:")
print("Game Type | Production Cost | Expected Profit")
print("-" * 45)
for row in game_data:
    print(f"{row[0]:<15} | ${int(row[1]):>10,} | ${int(row[2]):>12,}")

# Prepare features (Production Cost) and target (Profit)
X = game_data[:, 1].astype(float).reshape(-1, 1)  # Production costs
y = game_data[:, 2].astype(float)  # Expected profits

print(f"\nDataset shape: {X.shape}")
print(f"Production cost range: ${X.min():,.0f} - ${X.max():,.0f}")
print(f"Profit range: ${y.min():,.0f} - ${y.max():,.0f}")

# Split data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nData Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create Decision Tree Regressor with reduced complexity to prevent overfitting
regressor = DecisionTreeRegressor(
    random_state=42, 
    max_depth=4,           # Reduced from 10 to prevent overfitting
    min_samples_split=3,   # Require at least 3 samples to split a node
    min_samples_leaf=2     # Require at least 2 samples in each leaf
)

# Train model on training data only
regressor.fit(X_train, y_train)

# Make predictions on both training and test sets
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Evaluate model performance
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training MSE: ${train_mse:,.0f}")
print(f"Testing MSE: ${test_mse:,.0f}")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")

# Check for overfitting
r2_difference = train_r2 - test_r2
print(f"\nOverfitting Analysis:")
print(f"R² Difference (Train - Test): {r2_difference:.3f}")
if r2_difference > 0.2:
    print("⚠️ Model may be overfitting!")
elif r2_difference < 0:
    print("⚠️ Model may be underfitting!")
else:
    print("✅ Model appears to generalize well!")

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(regressor, X, y, cv=3, scoring='r2')
print(f"\nCross-Validation Results:")
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Make a prediction for a specific production cost
test_cost = 3750
prediction = regressor.predict([[test_cost]])
print(f"\nPrediction for ${test_cost:,} production cost: ${prediction[0]:,.0f} profit")

# Create smooth curve for visualization
X_min, X_max = float(X.min()), float(X.max())
X_grid = np.arange(X_min, X_max + 100, 100).reshape(-1, 1)
y_grid_pred = regressor.predict(X_grid)

# Visualization
plt.figure(figsize=(15, 10))

# Main plot: Model predictions
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, color='red', s=100, alpha=0.8, label='Training Data', edgecolors='black')
plt.scatter(X_test, y_test, color='orange', s=100, alpha=0.8, label='Test Data', edgecolors='black')
plt.plot(X_grid, y_grid_pred, color='blue', linewidth=2, label='Decision Tree Prediction')
plt.scatter([test_cost], prediction, color='green', s=200, marker='*', 
           label=f'Prediction: ${prediction[0]:,.0f}', edgecolors='black', zorder=5)
plt.title('Game Development: Profit vs Production Cost', fontweight='bold')
plt.xlabel('Production Cost ($)')
plt.ylabel('Expected Profit ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Training vs Test Performance
plt.subplot(2, 2, 2)
performance_data = ['Training', 'Testing']
r2_scores = [train_r2, test_r2]
mse_scores = [train_mse, test_mse]

bars = plt.bar(performance_data, r2_scores, color=['skyblue', 'lightcoral'], alpha=0.7, edgecolor='black')
plt.title('R² Score: Training vs Testing', fontweight='bold')
plt.ylabel('R² Score')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{r2_scores[i]:.3f}', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

# MSE Comparison
plt.subplot(2, 2, 3)
bars = plt.bar(performance_data, mse_scores, color=['lightgreen', 'lightpink'], alpha=0.7, edgecolor='black')
plt.title('MSE: Training vs Testing', fontweight='bold')
plt.ylabel('Mean Squared Error ($)')
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_scores)*0.02, 
             f'${mse_scores[i]:,.0f}', ha='center', fontweight='bold')
plt.grid(True, alpha=0.3)

# Cross-validation scores
plt.subplot(2, 2, 4)
plt.bar(range(len(cv_scores)), cv_scores, color='gold', alpha=0.7, edgecolor='black')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation R² Scores', fontweight='bold')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display feature importance
print(f"\nFeature Importance:")
print(f"Production Cost: {regressor.feature_importances_[0]:.3f}")

# Show additional predictions for different cost ranges
print(f"\nAdditional Predictions:")
test_costs = [2000, 5000, 10000, 20000]
for cost in test_costs:
    pred = regressor.predict([[cost]])
    print(f"${cost:,} production cost → ${pred[0]:,.0f} expected profit")

# Model complexity information
print(f"\nModel Complexity:")
print(f"Max depth: {regressor.max_depth}")
print(f"Min samples split: {regressor.min_samples_split}")
print(f"Min samples leaf: {regressor.min_samples_leaf}")
print(f"Number of leaves: {regressor.get_n_leaves()}")