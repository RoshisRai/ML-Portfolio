"""
Transportation Price Prediction using Linear Regression
=====================================================
Predicts transportation costs based on distance traveled.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample transportation data (distance in KM, price in Rs)
distances = np.array([15, 30, 45, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390]).reshape(-1, 1)
prices = np.array([90, 175, 260, 320, 450, 550, 630, 720, 780, 840, 870, 900, 950, 1020, 1090])

# Train model
model = LinearRegression()
model.fit(distances, prices)

# Make predictions
test_distances = np.array([45, 75, 120, 135, 270]).reshape(-1, 1)
predictions = model.predict(test_distances)

# Evaluate
r2 = model.score(distances, prices)
print(f"R² Score: {r2:.3f}")
print(f"Price per KM: Rs {model.coef_[0]:.2f}")

# Display predictions
for dist, price in zip(test_distances.flatten(), predictions):
    print(f"Distance: {dist} KM → Predicted Price: Rs {price:.0f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(distances, prices, color='red', label='Training Data', s=100)
plt.plot(distances, model.predict(distances), 'b-', label='Regression Line')
plt.scatter(test_distances, predictions, color='green', marker='^', s=100, label='Predictions')
plt.xlabel('Distance (KM)')
plt.ylabel('Price (Rs)')
plt.title('Transportation Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()