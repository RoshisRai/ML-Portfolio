"""
Support Vector Machine Classification for Breast Cancer Diagnosis
================================================================
This script demonstrates SVM classification to predict breast cancer diagnosis
(malignant vs benign) based on tumor characteristics. Includes kernel comparison,
hyperparameter tuning, and comprehensive evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load breast cancer dataset
cancer = load_breast_cancer()
print("Breast Cancer Dataset Overview:")
print(f"Features: {len(cancer.feature_names)} tumor characteristics")
print(f"Classes: {cancer.target_names}")
print(f"Dataset shape: {cancer.data.shape}")

# Create DataFrame for better handling
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

print(f"Class distribution: Malignant={np.sum(y==0)}, Benign={np.sum(y==1)}")

# Split data and scale features (important for SVM)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Compare different SVM kernels
kernels = ['linear', 'rbf', 'poly']
kernel_scores = []

print(f"\nKernel Comparison:")
print("Kernel | Train Acc | Test Acc | CV Score")
print("-" * 40)

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    
    kernel_scores.append((kernel, train_acc, test_acc, cv_scores.mean()))
    print(f"{kernel:6} | {train_acc:8.3f} | {test_acc:7.3f} | {cv_scores.mean():7.3f}")

# Select best kernel
best_kernel = max(kernel_scores, key=lambda x: x[3])[0]
print(f"\nBest kernel: {best_kernel}")

# Train final model with best kernel
svm_classifier = SVC(kernel=best_kernel, random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Evaluate model
y_pred_test = svm_classifier.predict(X_test_scaled)
train_accuracy = svm_classifier.score(X_train_scaled, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance ({best_kernel} kernel):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=cancer.target_names))

# Cross-validation for robust evaluation
cv_scores = cross_val_score(svm_classifier, X_train_scaled, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Prediction function
def predict_cancer_diagnosis(tumor_features):
    """Predict cancer diagnosis based on tumor characteristics"""
    # Note: In practice, you'd need all 30 features
    # This is simplified for demonstration
    features_scaled = scaler.transform([tumor_features])
    prediction = svm_classifier.predict(features_scaled)
    probabilities = svm_classifier.predict_proba(features_scaled) if hasattr(svm_classifier, "predict_proba") else None
    
    diagnosis = cancer.target_names[prediction[0]]
    print(f"Prediction: {diagnosis}")
    return prediction[0]

# Visualization
plt.figure(figsize=(12, 8))

# Kernel comparison
plt.subplot(2, 2, 1)
kernel_names = [item[0] for item in kernel_scores]
cv_scores_plot = [item[3] for item in kernel_scores]
bars = plt.bar(kernel_names, cv_scores_plot, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('SVM Kernel Comparison')
plt.ylabel('Cross-Validation Accuracy')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{cv_scores_plot[i]:.3f}', ha='center', fontweight='bold')

# Confusion Matrix
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
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

# Feature importance (for linear kernel)
if best_kernel == 'linear':
    feature_importance = np.abs(svm_classifier.coef_[0])
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features
    
    print(f"\nTop 10 Most Important Features (Linear SVM):")
    for i, idx in enumerate(reversed(top_features_idx)):
        print(f"{i+1:2d}. {cancer.feature_names[idx]:<25} {feature_importance[idx]:.4f}")

print(f"\nModel Info: Support Vector Machine with {best_kernel} kernel, StandardScaler applied")
print(f"Support vectors: {svm_classifier.n_support_}")