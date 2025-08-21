"""
Multinomial Naive Bayes Classification for Urban vs Rural Area Classification
============================================================================
This script demonstrates Multinomial Naive Bayes classification to predict 
urban vs rural areas based on feature counts (houses, cars, trees, etc.).
Includes comprehensive evaluation, feature analysis, and area classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Urban vs Rural area data with feature counts
print("Urban vs Rural Area Classification Dataset:")
print("Features represent counts of objects in different areas")
print("Classes: 0 = Rural, 1 = Urban")

# Training data: feature counts for different areas
training_data = [
    # Urban areas (class 1) - more houses, cars, shops, streets
    {'house': 100, 'street': 50, 'shop': 25, 'car': 100, 'tree': 20, 'river': 0},
    {'house': 120, 'street': 60, 'shop': 30, 'car': 90, 'tree': 15, 'river': 0},
    {'house': 80, 'street': 40, 'shop': 20, 'car': 110, 'tree': 25, 'river': 1},
    {'house': 90, 'street': 45, 'shop': 22, 'car': 95, 'tree': 18, 'river': 0},
    {'house': 110, 'street': 55, 'shop': 28, 'car': 105, 'tree': 22, 'river': 0},
    
    # Rural areas (class 0) - more trees, rivers, fewer urban features
    {'house': 5, 'street': 5, 'shop': 0, 'car': 10, 'tree': 500, 'river': 3},
    {'house': 8, 'street': 3, 'shop': 1, 'car': 12, 'tree': 480, 'river': 2},
    {'house': 3, 'street': 2, 'shop': 0, 'car': 8, 'tree': 520, 'river': 4},
    {'house': 6, 'street': 4, 'shop': 1, 'car': 9, 'tree': 490, 'river': 3},
    {'house': 4, 'street': 3, 'shop': 0, 'car': 11, 'tree': 510, 'river': 2}
]

# Labels: 1 = Urban (first 5), 0 = Rural (last 5)
training_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Transform dictionaries to feature vectors
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(training_data)
y_train = training_labels

print(f"\nTraining Data Overview:")
print(f"Features: {dv.feature_names_}")
print(f"Dataset shape: {X_train.shape}")
print(f"Class distribution: Urban={np.sum(y_train==1)}, Rural={np.sum(y_train==0)}")

# Display training data in readable format
train_df = pd.DataFrame(X_train, columns=dv.feature_names_)
train_df['area_type'] = ['Urban' if label == 1 else 'Rural' for label in y_train]
print(f"\nTraining Dataset:")
print(train_df.to_string(index=False))

# Test data for evaluation
test_data = [
    # Should be classified as Urban
    {'house': 85, 'street': 42, 'shop': 18, 'car': 88, 'tree': 28, 'river': 0},
    {'house': 95, 'street': 48, 'shop': 24, 'car': 102, 'tree': 16, 'river': 1},
    
    # Should be classified as Rural  
    {'house': 7, 'street': 4, 'shop': 0, 'car': 9, 'tree': 485, 'river': 3},
    {'house': 4, 'street': 2, 'shop': 1, 'car': 7, 'tree': 505, 'river': 4}
]

test_labels = np.array([1, 1, 0, 0])  # Expected classifications

# Transform test data
X_test = dv.transform(test_data)
y_test = test_labels

# Train Multinomial Naive Bayes with different alpha values
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
alpha_results = []

print(f"\nAlpha Parameter Comparison:")
print("Alpha | Train Acc | Test Acc")
print("-" * 30)

for alpha in alpha_values:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    
    train_acc = mnb.score(X_train, y_train)
    test_acc = mnb.score(X_test, y_test)
    
    alpha_results.append((alpha, mnb, train_acc, test_acc))
    print(f"{alpha:5.1f} | {train_acc:8.3f} | {test_acc:7.3f}")

# Select best alpha (highest test accuracy)
optimal_alpha, best_model, _, _ = max(alpha_results, key=lambda x: x[3])
print(f"\nOptimal alpha: {optimal_alpha}")

# Evaluate best model
y_pred = best_model.predict(X_test)
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance (alpha={optimal_alpha}):")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")

# Classification report
class_names = ['Rural', 'Urban']
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Feature importance analysis
feature_log_probs = best_model.feature_log_prob_
print(f"\nFeature Importance Analysis:")
print("Feature importance based on log probabilities for each class")
print("\nRural Areas (Class 0) - Top features:")
rural_importance = feature_log_probs[0]
rural_top_indices = np.argsort(rural_importance)[-3:][::-1]
for i, idx in enumerate(rural_top_indices):
    print(f"{i+1}. {dv.feature_names_[idx]}: {rural_importance[idx]:.3f}")

print(f"\nUrban Areas (Class 1) - Top features:")
urban_importance = feature_log_probs[1]
urban_top_indices = np.argsort(urban_importance)[-3:][::-1]
for i, idx in enumerate(urban_top_indices):
    print(f"{i+1}. {dv.feature_names_[idx]}: {urban_importance[idx]:.3f}")

# Prediction function
def classify_area(feature_counts):
    """Classify area as Urban or Rural based on feature counts"""
    if not isinstance(feature_counts, dict):
        print("Error: Input must be a dictionary with feature counts")
        return None
    
    # Transform input
    input_vector = dv.transform([feature_counts])
    prediction = best_model.predict(input_vector)
    probabilities = best_model.predict_proba(input_vector)
    
    area_type = class_names[prediction[0]]
    confidence = max(probabilities[0]) * 100
    
    print(f"Input features: {feature_counts}")
    print(f"Prediction: {area_type} (Confidence: {confidence:.1f}%)")
    print(f"Probabilities: Rural={probabilities[0][0]:.3f}, Urban={probabilities[0][1]:.3f}")
    
    return prediction[0]

# Visualization
plt.figure(figsize=(15, 10))

# Alpha parameter comparison
plt.subplot(2, 3, 1)
alphas = [result[0] for result in alpha_results]
test_accs = [result[3] for result in alpha_results]
plt.plot(alphas, test_accs, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.8, label=f'Optimal α={optimal_alpha}')
plt.title('Alpha Parameter Tuning')
plt.xlabel('Alpha (Smoothing Parameter)')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature distribution by class
plt.subplot(2, 3, 2)
feature_means_rural = np.mean(X_train[y_train == 0], axis=0)
feature_means_urban = np.mean(X_train[y_train == 1], axis=0)

x_pos = np.arange(len(dv.feature_names_))
width = 0.35

plt.bar(x_pos - width/2, feature_means_rural, width, label='Rural', color='lightgreen', alpha=0.7)
plt.bar(x_pos + width/2, feature_means_urban, width, label='Urban', color='lightcoral', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Average Count')
plt.title('Average Feature Counts by Area Type')
plt.xticks(x_pos, [name.replace('_', '\n') for name in dv.feature_names_], rotation=45)
plt.legend()

# Confusion Matrix
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Performance comparison
plt.subplot(2, 3, 4)
plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], 
        color=['lightgreen', 'lightcoral'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)

# Feature importance heatmap
plt.subplot(2, 3, 5)
importance_matrix = np.abs(feature_log_probs)
sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=dv.feature_names_, yticklabels=class_names)
plt.title('Feature Importance Heatmap')
plt.xlabel('Features')
plt.ylabel('Area Type')

# Test predictions visualization
plt.subplot(2, 3, 6)
test_df = pd.DataFrame(X_test, columns=dv.feature_names_)
test_df['actual'] = ['Urban' if label == 1 else 'Rural' for label in y_test]
test_df['predicted'] = ['Urban' if pred == 1 else 'Rural' for pred in y_pred]
test_df['correct'] = test_df['actual'] == test_df['predicted']

colors = ['lightcoral' if not correct else 'lightgreen' for correct in test_df['correct']]
plt.bar(range(len(test_df)), [1]*len(test_df), color=colors)
plt.title('Test Predictions')
plt.xlabel('Test Sample')
plt.ylabel('Prediction Result')
plt.xticks(range(len(test_df)), [f'Sample {i+1}' for i in range(len(test_df))])

# Add legend
import matplotlib.patches as patches
correct_patch = patches.Patch(color='lightgreen', label='Correct')
incorrect_patch = patches.Patch(color='lightcoral', label='Incorrect')
plt.legend(handles=[correct_patch, incorrect_patch])

plt.tight_layout()
plt.show()

# Example predictions
print(f"\nExample Predictions:")
print("=" * 60)

# New test cases
new_areas = [
    {'house': 75, 'street': 35, 'shop': 15, 'car': 80, 'tree': 30, 'river': 1},  # Should be Urban
    {'house': 6, 'street': 3, 'shop': 0, 'car': 8, 'tree': 450, 'river': 5},     # Should be Rural
    {'house': 50, 'street': 25, 'shop': 10, 'car': 60, 'tree': 100, 'river': 2}  # Mixed case
]

for i, area in enumerate(new_areas):
    print(f"\nTest Area {i+1}:")
    classify_area(area)
    print("-" * 50)

print(f"\nModel Information:")
print(f"Algorithm: Multinomial Naive Bayes")
print(f"Optimal alpha (smoothing): {optimal_alpha}")
print(f"Features: {', '.join(dv.feature_names_)}")
print(f"Classes: Rural (0), Urban (1)")

print(f"\nModel Insights:")
print(f"• Multinomial NB works well with count/frequency data")
print(f"• Feature independence assumed (each object type treated separately)")
print(f"• Alpha smoothing prevents zero probabilities for unseen feature combinations")
print(f"• Model learns probability distributions for each area type and feature")