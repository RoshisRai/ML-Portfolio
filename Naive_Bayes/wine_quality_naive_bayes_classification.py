"""
Gaussian Naive Bayes Classification for Wine Quality Prediction
==============================================================
This script demonstrates Gaussian Naive Bayes classification to predict wine
quality/origin based on chemical characteristics. Includes feature analysis,
comprehensive evaluation, and wine quality interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load wine dataset
wine = load_wine()
print("Wine Dataset Overview:")
print(f"Features: {len(wine.feature_names)} chemical characteristics")
print(f"Classes: {wine.target_names}")
print(f"Dataset shape: {wine.data.shape}")

# Create DataFrame for better handling
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print(f"\nClass distribution: {np.bincount(y)}")
print(f"Feature ranges:")
print(f"Min values: {X.min().min():.3f}")
print(f"Max values: {X.max().max():.3f}")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Compare scaled vs unscaled performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models (scaled and unscaled)
gnb_unscaled = GaussianNB()
gnb_scaled = GaussianNB()

gnb_unscaled.fit(X_train, y_train)
gnb_scaled.fit(X_train_scaled, y_train)

# Evaluate both versions
unscaled_train_acc = gnb_unscaled.score(X_train, y_train)
unscaled_test_acc = gnb_unscaled.score(X_test, y_test)
scaled_train_acc = gnb_scaled.score(X_train_scaled, y_train)
scaled_test_acc = gnb_scaled.score(X_test_scaled, y_test)

print(f"\nModel Comparison:")
print("Version     | Train Acc | Test Acc")
print("-" * 35)
print(f"Unscaled    | {unscaled_train_acc:8.3f} | {unscaled_test_acc:7.3f}")
print(f"Scaled      | {scaled_train_acc:8.3f} | {scaled_test_acc:7.3f}")

# Select better performing model
if scaled_test_acc > unscaled_test_acc:
    best_model = gnb_scaled
    best_name = "Scaled"
    X_train_best = X_train_scaled
    X_test_best = X_test_scaled
else:
    best_model = gnb_unscaled
    best_name = "Unscaled"
    X_train_best = X_train
    X_test_best = X_test

print(f"\nBest model: {best_name} Gaussian Naive Bayes")

# Evaluate best model
y_pred_test = best_model.predict(X_test_best)
train_accuracy = best_model.score(X_train_best, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance ({best_name}):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=wine.target_names))

# Cross-validation for robust evaluation
cv_scores = cross_val_score(best_model, X_train_best, y_train, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance analysis (based on variance in each class)
print(f"\nTop 10 Most Discriminative Features:")
feature_importance = []
for i, feature in enumerate(wine.feature_names):
    class_variances = []
    for class_idx in range(len(wine.target_names)):
        class_mask = (y_train == class_idx)
        if best_name == "Scaled":
            class_variance = np.var(X_train_scaled[class_mask, i])
        else:
            class_variance = np.var(X_train.iloc[class_mask, i])
        class_variances.append(class_variance)
    
    # Use variance difference as importance measure
    importance = np.std(class_variances)
    feature_importance.append((feature, importance))

# Sort by importance
feature_importance.sort(key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(feature_importance[:10]):
    print(f"{i+1:2d}. {feature:<25} {importance:.4f}")

# Prediction function
def predict_wine_quality(chemical_characteristics):
    """Predict wine quality/origin based on chemical characteristics"""
    if len(chemical_characteristics) != 13:
        print("Error: Need exactly 13 chemical measurements")
        return None
    
    input_data = np.array(chemical_characteristics).reshape(1, -1)
    
    if best_name == "Scaled":
        input_data = scaler.transform(input_data)
    
    prediction = best_model.predict(input_data)
    probabilities = best_model.predict_proba(input_data)
    
    wine_class = wine.target_names[prediction[0]]
    confidence = max(probabilities[0]) * 100
    
    print(f"Predicted wine class: {wine_class} (Confidence: {confidence:.1f}%)")
    print(f"Class probabilities:")
    for i, prob in enumerate(probabilities[0]):
        print(f"  {wine.target_names[i]}: {prob:.3f}")
    
    return prediction[0]

# Visualization
plt.figure(figsize=(15, 10))

# Scaled vs Unscaled comparison
plt.subplot(2, 3, 1)
models = ['Unscaled', 'Scaled']
test_accuracies = [unscaled_test_acc, scaled_test_acc]
bars = plt.bar(models, test_accuracies, color=['lightcoral', 'lightgreen'])
plt.title('Scaled vs Unscaled Performance')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{test_accuracies[i]:.3f}', ha='center', fontweight='bold')

# Class distribution
plt.subplot(2, 3, 2)
class_counts = np.bincount(y)
colors = ['lightblue', 'lightgreen', 'lightcoral']
plt.bar(wine.target_names, class_counts, color=colors)
plt.title('Wine Class Distribution')
plt.ylabel('Count')
plt.xticks(rotation=45)

# Confusion Matrix
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Wine Class')
plt.xlabel('Predicted Wine Class')

# Performance comparison
plt.subplot(2, 3, 4)
plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], 
        color=['lightgreen', 'lightcoral'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Cross-validation scores
plt.subplot(2, 3, 5)
plt.bar(range(len(cv_scores)), cv_scores, color='gold')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# Top features importance
plt.subplot(2, 3, 6)
top_features = [item[0] for item in feature_importance[:8]]
top_importance = [item[1] for item in feature_importance[:8]]
plt.barh(range(len(top_features)), top_importance, color='purple', alpha=0.7)
plt.yticks(range(len(top_features)), [f.replace('_', ' ').title() for f in top_features])
plt.title('Top 8 Discriminative Features')
plt.xlabel('Importance Score')

plt.tight_layout()
plt.show()

# Example prediction with first sample
print(f"\nExample Prediction:")
sample_idx = 0
sample_data = wine.data[sample_idx]
actual_class = wine.target_names[wine.target[sample_idx]]

print(f"Sample chemical profile: {sample_data[:5]}... (showing first 5 features)")
print(f"Actual wine class: {actual_class}")
predicted_class = predict_wine_quality(sample_data)

print(f"\nModel Info: {best_name} Gaussian Naive Bayes")
print(f"Chemical features: 13 characteristics (alcohol, acidity, phenols, etc.)")
print(f"Wine classes: {', '.join(wine.target_names)}")

# Model assumptions and insights
print(f"\nModel Insights:")
print(f"• Gaussian Naive Bayes assumes features follow normal distribution")
print(f"• Feature independence assumed (chemical characteristics treated separately)")
print(f"• {best_name.lower()} features performed better for this wine dataset")
print(f"• Model learns probability distributions for each wine class and chemical feature")