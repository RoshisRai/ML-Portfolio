"""
Multinomial Naive Bayes Classification for Handwritten Digit Recognition
========================================================================
This script demonstrates Multinomial Naive Bayes classification to predict 
handwritten digits (0-9) based on pixel count features. Includes alpha 
smoothing comparison, comprehensive evaluation, and digit recognition analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load handwritten digits dataset
digits = load_digits()
print("Handwritten Digits Dataset Overview:")
print(f"Images: 8x8 pixel grayscale handwritten digits")
print(f"Classes: {digits.target_names} (digits 0-9)")
print(f"Dataset shape: {digits.data.shape}")
print(f"Feature range: {digits.data.min():.1f} - {digits.data.max():.1f} (pixel intensities)")

# Create DataFrame for better handling
X = pd.DataFrame(digits.data)
y = digits.target

print(f"Class distribution: {np.bincount(y)}")

# Split data with stratification and fixed random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Compare different alpha (smoothing) values
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
alpha_results = []

print(f"\nAlpha (Smoothing) Parameter Comparison:")
print("Alpha | Train Acc | Test Acc | CV Score")
print("-" * 40)

for alpha in alpha_values:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    
    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)
    cv_scores = cross_val_score(mnb, X_train, y_train, cv=5)
    
    alpha_results.append((alpha, train_acc, test_acc, cv_scores.mean()))
    print(f"{alpha:5.1f} | {train_acc:8.3f} | {test_acc:7.3f} | {cv_scores.mean():7.3f}")

# Select optimal alpha
optimal_alpha = max(alpha_results, key=lambda x: x[3])[0]
print(f"\nOptimal alpha: {optimal_alpha}")

# Train final model with optimal alpha
mnb_classifier = MultinomialNB(alpha=optimal_alpha)
mnb_classifier.fit(X_train, y_train)

# Evaluate model
y_pred_test = mnb_classifier.predict(X_test)
train_accuracy = mnb_classifier.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance (alpha={optimal_alpha}):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Cross-validation for robust evaluation
cv_scores = cross_val_score(mnb_classifier, X_train, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance analysis (log probabilities)
feature_log_probs = mnb_classifier.feature_log_prob_
print(f"\nMost Important Pixels for Each Digit:")
for digit in range(10):
    top_pixels = np.argsort(feature_log_probs[digit])[-5:]  # Top 5 pixels
    print(f"Digit {digit}: Pixels {top_pixels} (positions in 8x8 grid)")

# Prediction function
def predict_digit(pixel_values):
    """Predict handwritten digit based on 8x8 pixel values"""
    if len(pixel_values) != 64:
        print("Error: Need exactly 64 pixel values (8x8 image)")
        return None
    
    input_data = np.array(pixel_values).reshape(1, -1)
    prediction = mnb_classifier.predict(input_data)
    probabilities = mnb_classifier.predict_proba(input_data)
    
    digit = prediction[0]
    confidence = max(probabilities[0]) * 100
    
    print(f"Predicted digit: {digit} (Confidence: {confidence:.1f}%)")
    print("Top 3 predictions:")
    top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
    for i, idx in enumerate(top_3_indices):
        print(f"  {i+1}. Digit {idx}: {probabilities[0][idx]:.3f}")
    
    return digit

# Visualization
plt.figure(figsize=(15, 10))

# Alpha parameter comparison
plt.subplot(2, 3, 1)
alphas = [result[0] for result in alpha_results]
cv_scores_plot = [result[3] for result in alpha_results]
plt.plot(alphas, cv_scores_plot, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.8, label=f'Optimal α={optimal_alpha}')
plt.title('Alpha Parameter Tuning')
plt.xlabel('Alpha (Smoothing Parameter)')
plt.ylabel('Cross-Validation Accuracy')
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

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
sample_images = digits.images[:10]
for i in range(10):
    plt.subplot(4, 5, i+11)  # Position in a 4x5 grid within the subplot
    plt.imshow(sample_images[i], cmap='gray')
    plt.title(f'Digit: {digits.target[i]}', fontsize=8)
    plt.axis('off')
plt.suptitle('Sample Handwritten Digits', y=0.25)

# Feature importance heatmap (average across all digits)
plt.subplot(2, 3, 6)
avg_importance = np.mean(feature_log_probs, axis=0)
importance_image = avg_importance.reshape(8, 8)
plt.imshow(importance_image, cmap='hot')
plt.title('Average Feature Importance\n(Pixel Significance)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Example predictions with sample digits
print(f"\nExample Predictions:")
sample_indices = [0, 10, 20, 30, 40]
for idx in sample_indices:
    actual_digit = digits.target[idx]
    pixel_data = digits.data[idx]
    print(f"\nSample {idx}: Actual digit = {actual_digit}")
    predicted = predict_digit(pixel_data)
    print("-" * 50)

# Model insights
print(f"\nModel Information:")
print(f"Algorithm: Multinomial Naive Bayes")
print(f"Optimal alpha (smoothing): {optimal_alpha}")
print(f"Feature type: Pixel intensity counts (0-16)")
print(f"Classes: 10 digits (0-9)")

print(f"\nMultinomial NB Insights:")
print(f"• Assumes features represent counts or frequencies (pixel intensities)")
print(f"• Alpha smoothing prevents zero probabilities in sparse data")
print(f"• Works well with discrete/count data like pixel intensities")
print(f"• Feature independence assumed (each pixel treated separately)")
print(f"• Optimal alpha={optimal_alpha} balances overfitting vs underfitting")

# Per-class performance analysis
print(f"\nPer-Digit Accuracy Analysis:")
for digit in range(10):
    digit_mask = (y_test == digit)
    if np.any(digit_mask):
        digit_predictions = y_pred_test[digit_mask]
        digit_accuracy = np.mean(digit_predictions == digit)
        print(f"Digit {digit}: {digit_accuracy:.3f} accuracy ({np.sum(digit_mask)} samples)")