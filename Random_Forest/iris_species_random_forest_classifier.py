"""
Random Forest Classification for Iris Species Prediction
=======================================================
This script demonstrates multi-class classification using Random Forest
to predict iris flower species based on sepal and petal measurements.
Includes feature importance analysis and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and explore the iris dataset
iris = load_iris()
print("Iris Dataset Overview:")
print(f"Target species: {iris.target_names}")
print(f"Features: {iris.feature_names}")
print(f"Dataset shape: {iris.data.shape}")

# Create DataFrame for better data handling
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = iris.target

print(f"\nClass distribution: {np.bincount(y)}")

# Split data into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Create and train Random Forest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    oob_score=True  # Fixed: Enable out-of-bag scoring
)

rf_classifier.fit(X_train, y_train)

# Evaluate model performance
y_pred_test = rf_classifier.predict(X_test)
train_accuracy = rf_classifier.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=iris.target_names))

# Cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Prediction function
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """Predict iris species based on flower measurements"""
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    prediction = rf_classifier.predict(input_data)
    probability = rf_classifier.predict_proba(input_data)
    
    species_name = iris.target_names[prediction[0]]
    confidence = max(probability[0]) * 100
    
    print(f"Prediction: {species_name} (Confidence: {confidence:.1f}%)")
    return prediction[0]

# Visualization
plt.figure(figsize=(12, 8))

# Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')

# Feature Importance
plt.subplot(2, 2, 2)
sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
plt.title('Feature Importance')

# Training vs Test Accuracy
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

# Model information
print(f"\nModel Information:")
print(f"Number of estimators: {rf_classifier.n_estimators}")
print(f"Out-of-bag score: {rf_classifier.oob_score_:.3f}")