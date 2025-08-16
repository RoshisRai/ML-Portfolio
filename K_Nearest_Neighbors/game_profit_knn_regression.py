"""
K-Nearest Neighbors Regression for Game Development Profit Prediction
====================================================================
This script demonstrates KNN regression to predict game development profits
based on production costs. Includes hyperparameter tuning and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# Prepare features and target
X = game_data[:, 1:2].astype(float)  # Production costs
y = game_data[:, 2].astype(float)    # Expected profits

print(f"\nDataset shape: {X.shape}")
print(f"Cost range: ${X.min():,.0f} - ${X.max():,.0f}")

# Split data and scale features (important for KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find optimal k value through cross-validation
k_range = range(1, 8)  # Reasonable range for small dataset
k_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=3, scoring='r2')
    k_scores.append(cv_scores.mean())

optimal_k = k_range[np.argmax(k_scores)]
print(f"\nOptimal K: {optimal_k}")

# Train final model
knn_regressor = KNeighborsRegressor(n_neighbors=optimal_k)
knn_regressor.fit(X_train_scaled, y_train)

# Evaluate model
y_pred_test = knn_regressor.predict(X_test_scaled)
train_r2 = knn_regressor.score(X_train_scaled, y_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nModel Performance (K={optimal_k}):")
print(f"Training R²: {train_r2:.3f}")
print(f"Testing R²: {test_r2:.3f}")
print("✅ Model generalizes well!" if train_r2 - test_r2 <= 0.2 else "⚠️ Model may be overfitting!")

# Prediction function
def predict_game_profit(production_cost):
    """Predict game profit based on production cost"""
    cost_scaled = scaler.transform([[production_cost]])
    prediction = knn_regressor.predict(cost_scaled)
    print(f"${production_cost:,} → ${prediction[0]:,.0f} profit")
    return prediction[0]

# Visualization
plt.figure(figsize=(12, 8))

# Main prediction plot
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, color='red', s=100, alpha=0.8, label='Training', edgecolors='black')
plt.scatter(X_test, y_test, color='orange', s=100, alpha=0.8, label='Test', edgecolors='black')

# Create prediction curve
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_plot_pred = knn_regressor.predict(X_plot_scaled)
plt.plot(X_plot, y_plot_pred, color='blue', linewidth=2, label=f'KNN (K={optimal_k})')

plt.title('Game Profit vs Production Cost')
plt.xlabel('Production Cost ($)')
plt.ylabel('Expected Profit ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# K selection plot
plt.subplot(2, 2, 2)
plt.plot(k_range, k_scores, 'bo-', linewidth=2)
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
plt.title('K Selection via Cross-Validation')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('CV R² Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Performance comparison
plt.subplot(2, 2, 3)
plt.bar(['Training', 'Testing'], [train_r2, test_r2], color=['lightgreen', 'lightcoral'])
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)

# Actual vs Predicted
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_test, alpha=0.7, edgecolors='black')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted')
plt.xlabel('Actual Profit ($)')
plt.ylabel('Predicted Profit ($)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Example predictions
print(f"\nExample Predictions:")
test_costs = [2000, 5000, 10000, 20000]
for cost in test_costs:
    predict_game_profit(cost)

print(f"\nModel Info: K-Nearest Neighbors with K={optimal_k}, StandardScaler applied")