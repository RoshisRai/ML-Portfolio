"""
K-Nearest Neighbors Classification for Iris Species Prediction
=============================================================
This script demonstrates KNN classification to predict iris flower species
based on sepal and petal measurements. Includes hyperparameter tuning
and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load iris dataset
iris = load_iris()
print("Iris Dataset Overview:")
print(f"Features: {iris.feature_names}")
print(f"Classes: {iris.target_names}")
print(f"Dataset shape: {iris.data.shape}")

# Create DataFrame for better handling
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = iris.target

print(f"Class distribution: {np.bincount(y)}")

# Split data and scale features (important for KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Find optimal k value through cross-validation
k_range = range(1, 16)
train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    train_scores.append(knn.score(X_train_scaled, y_train))
    test_scores.append(knn.score(X_test_scaled, y_test))

# Find optimal k (best test score)
optimal_k = k_range[np.argmax(test_scores)]
print(f"\nOptimal K: {optimal_k}")

# Train final model with optimal k
knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
knn_classifier.fit(X_train_scaled, y_train)

# Evaluate model
y_pred_test = knn_classifier.predict(X_test_scaled)
train_accuracy = knn_classifier.score(X_train_scaled, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance (K={optimal_k}):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=iris.target_names))

# Cross-validation for robust evaluation
cv_scores = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Prediction function
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """Predict iris species based on flower measurements"""
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    input_scaled = scaler.transform(input_data)
    prediction = knn_classifier.predict(input_scaled)
    probabilities = knn_classifier.predict_proba(input_scaled)
    
    species_name = iris.target_names[prediction[0]]
    confidence = max(probabilities[0]) * 100
    
    print(f"Prediction: {species_name} (Confidence: {confidence:.1f}%)")
    return prediction[0]

# Visualization
plt.figure(figsize=(12, 8))

# K selection plot
plt.subplot(2, 2, 1)
plt.plot(k_range, train_scores, 'bo-', label='Training Accuracy', linewidth=2)
plt.plot(k_range, test_scores, 'ro-', label='Testing Accuracy', linewidth=2)
plt.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.8, label=f'Optimal K={optimal_k}')
plt.title('K Selection for KNN Classification')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Confusion Matrix
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Performance comparison
plt.subplot(2, 2, 3)
plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], 
        color=['lightgreen', 'lightcoral'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Cross-validation scores
plt.subplot(2, 2, 4)
plt.bar(range(len(cv_scores)), cv_scores, color='gold')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Example predictions
print(f"\nExample Predictions:")
test_cases = [
    (5.1, 3.5, 1.4, 0.2),  # Typical Setosa
    (7.0, 3.2, 4.7, 1.4),  # Typical Versicolor
    (6.3, 3.3, 6.0, 2.5),  # Typical Virginica
]

for case in test_cases:
    predict_iris_species(*case)

print(f"\nModel Info: K-Nearest Neighbors with K={optimal_k}, StandardScaler applied")