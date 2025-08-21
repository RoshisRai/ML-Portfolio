"""
Naive Bayes Classification for Handwritten Digit Recognition
===========================================================
This script demonstrates Naive Bayes classification to predict handwritten digits
(0-9) based on pixel intensity values. Includes Naive Bayes variant comparison,
feature binarization, and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load handwritten digits dataset
digits = load_digits()
print("Handwritten Digits Dataset Overview:")
print(f"Images: 8x8 pixel grayscale handwritten digits")
print(f"Classes: {digits.target_names} (digits 0-9)")
print(f"Dataset shape: {digits.data.shape}")
print(f"Feature range: {digits.data.min():.1f} - {digits.data.max():.1f}")

# Create DataFrame for better handling
X = pd.DataFrame(digits.data)
y = digits.target

print(f"Class distribution: {np.bincount(y)}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Prepare data for different Naive Bayes variants
# Binarize for Bernoulli NB (convert to 0/1)
X_train_binary = (X_train > 8).astype(int)  # Threshold for binarization
X_test_binary = (X_test > 8).astype(int)

# Scale for Gaussian NB
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compare different Naive Bayes variants
nb_variants = [
    ('Bernoulli', BernoulliNB(), X_train_binary, X_test_binary),
    ('Gaussian', GaussianNB(), X_train_scaled, X_test_scaled),
    ('Multinomial', MultinomialNB(), X_train, X_test)  # Raw features (non-negative)
]

nb_results = []
print(f"\nNaive Bayes Variant Comparison:")
print("Variant     | Train Acc | Test Acc | CV Score")
print("-" * 45)

for name, model, X_tr, X_te in nb_variants:
    model.fit(X_tr, y_train)
    
    train_acc = model.score(X_tr, y_train)
    test_acc = model.score(X_te, y_test)
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5)
    
    nb_results.append((name, model, train_acc, test_acc, cv_scores.mean(), X_tr, X_te))
    print(f"{name:11} | {train_acc:8.3f} | {test_acc:7.3f} | {cv_scores.mean():7.3f}")

# Select best variant
best_variant = max(nb_results, key=lambda x: x[4])
best_name, best_model, _, _, _, best_X_train, best_X_test = best_variant
print(f"\nBest variant: {best_name} Naive Bayes")

# Evaluate best model
y_pred_test = best_model.predict(best_X_test)
train_accuracy = best_model.score(best_X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance ({best_name} NB):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Cross-validation for robust evaluation
cv_scores = cross_val_score(best_model, best_X_train, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Prediction function
def predict_digit(pixel_values):
    """Predict handwritten digit based on 8x8 pixel values"""
    # Convert input based on best model type
    if best_name == 'Bernoulli':
        input_data = (np.array(pixel_values) > 8).astype(int).reshape(1, -1)
    elif best_name == 'Gaussian':
        input_data = scaler.transform(np.array(pixel_values).reshape(1, -1))
    else:  # Multinomial
        input_data = np.array(pixel_values).reshape(1, -1)
    
    prediction = best_model.predict(input_data)
    probabilities = best_model.predict_proba(input_data)
    
    digit = prediction[0]
    confidence = max(probabilities[0]) * 100
    
    print(f"Predicted digit: {digit} (Confidence: {confidence:.1f}%)")
    return digit

# Visualization
plt.figure(figsize=(15, 10))

# Naive Bayes variant comparison
plt.subplot(2, 3, 1)
variant_names = [result[0] for result in nb_results]
cv_scores_plot = [result[4] for result in nb_results]
bars = plt.bar(variant_names, cv_scores_plot, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Naive Bayes Variant Comparison')
plt.ylabel('Cross-Validation Accuracy')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{cv_scores_plot[i]:.3f}', ha='center', fontweight='bold')

# Confusion Matrix
plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Digit')
plt.xlabel('Predicted Digit')

# Performance comparison
plt.subplot(2, 3, 3)
plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], 
        color=['lightgreen', 'lightcoral'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Cross-validation scores
plt.subplot(2, 3, 4)
plt.bar(range(len(cv_scores)), cv_scores, color='gold')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# Sample digit images
plt.subplot(2, 3, 5)
fig_sample = plt.figure(figsize=(8, 2))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f'Digit: {digits.target[i]}')
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits')
plt.close(fig_sample)

# Feature importance visualization (for Bernoulli NB)
if best_name == 'Bernoulli':
    plt.subplot(2, 3, 6)
    feature_log_prob = best_model.feature_log_prob_
    avg_importance = np.mean(np.abs(feature_log_prob), axis=0)
    importance_image = avg_importance.reshape(8, 8)
    plt.imshow(importance_image, cmap='hot')
    plt.title('Feature Importance Heatmap')
    plt.colorbar()
else:
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, f'{best_name} NB\nSelected', 
             ha='center', va='center', fontsize=16, 
             transform=plt.gca().transAxes)
    plt.title('Best Model Selected')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Example predictions with sample digits
print(f"\nExample Predictions:")
sample_indices = [0, 1, 2, 3, 4]
for idx in sample_indices:
    actual_digit = digits.target[idx]
    pixel_data = digits.data[idx]
    predicted = predict_digit(pixel_data)
    print(f"Sample {idx}: Actual={actual_digit}, Predicted={predicted}")

print(f"\nModel Info: {best_name} Naive Bayes")
if best_name == 'Bernoulli':
    print("Features: Binarized pixel intensities (threshold=8)")
elif best_name == 'Gaussian':
    print("Features: Standardized pixel intensities")
else:
    print("Features: Raw pixel intensities (0-16)")