"""
Logistic Regression Implementation for Student Performance Classification
========================================================================
This script demonstrates binary classification using scikit-learn Logistic Regression
to predict student pass/fail outcomes based on academic performance metrics.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Create sample student performance data
data = {
    'study_hours_weekly': [12, 4, 18, 8, 2, 15, 10, 3, 20, 6, 14, 5, 16, 9, 1, 
                          17, 7, 13, 11, 0, 19, 8, 12, 4, 15, 6, 18, 3, 14, 9],
    
    'attendance_percentage': [95, 60, 98, 75, 45, 92, 85, 50, 100, 65, 90, 55, 
                             88, 80, 40, 94, 70, 87, 82, 35, 96, 77, 89, 58, 91, 
                             68, 97, 48, 86, 79],
    
    'previous_gpa': [3.8, 2.1, 3.9, 2.8, 1.5, 3.6, 3.2, 1.8, 4.0, 2.4, 3.5, 
                    2.0, 3.7, 3.0, 1.2, 3.8, 2.6, 3.4, 3.1, 1.0, 3.9, 2.7, 
                    3.3, 2.2, 3.6, 2.5, 3.8, 1.9, 3.4, 2.9],
    
    'assignment_completion': [98, 65, 100, 80, 50, 95, 88, 55, 100, 70, 92, 60, 
                             94, 85, 45, 96, 75, 90, 87, 40, 98, 78, 91, 62, 93, 
                             72, 97, 52, 89, 83],
    
    'participation_score': [9, 4, 10, 6, 2, 8, 7, 3, 10, 5, 8, 4, 9, 7, 1, 9, 
                           5, 8, 7, 2, 10, 6, 8, 4, 9, 5, 10, 3, 8, 6],
    
    'family_support': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 
                      1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    
    'extracurricular': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 
                       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    
    'Pass_Or_Fail': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 
                    1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Class distribution:\n{df['Pass_Or_Fail'].value_counts()}")

# Prepare features and target
X = df.drop('Pass_Or_Fail', axis=1)
y = df['Pass_Or_Fail']

# Split data (75/25) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train logistic regression model
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': log_reg.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nFeature Importance:")
print(feature_importance)

# Visualization
plt.figure(figsize=(15, 5))

# Confusion Matrix
plt.subplot(1, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix')

# Feature Importance
plt.subplot(1, 3, 2)
plt.barh(feature_importance['feature'], abs(feature_importance['coefficient']))
plt.title('Feature Importance')
plt.xlabel('|Coefficient|')

# Prediction Probabilities
plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[:, 1], bins=10, alpha=0.7, edgecolor='black')
plt.title('Prediction Probabilities')
plt.xlabel('P(Pass)')

plt.tight_layout()
plt.show()