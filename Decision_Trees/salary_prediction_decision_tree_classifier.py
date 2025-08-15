"""
Decision Tree Classification for Salary Prediction
=================================================
This script demonstrates binary classification using Decision Tree
to predict whether an employee's salary is above $100k based on
company, job title, and degree level.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and explore the dataset
df = pd.read_csv('Salarydata.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass Distribution:")
print(df['Salary_more_than_100k'].value_counts())

# Prepare and encode features
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

X = pd.DataFrame({
    'company': le_company.fit_transform(df['Company']),
    'job': le_job.fit_transform(df['Job']),
    'degree': le_degree.fit_transform(df['degree'])
})
y = df['Salary_more_than_100k']

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(
    random_state=42, max_depth=5, min_samples_split=5, min_samples_leaf=2
)
model.fit(X_train, y_train)

# Evaluate model
y_pred_test = model.predict(X_test)
train_accuracy = model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")
print("✅ Model generalizes well!" if train_accuracy - test_accuracy <= 0.1 else "⚠️ Model may be overfitting!")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Feature importance
feature_names = ['Company', 'Job', 'Degree']
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:")
print(feature_importance)

# Visualization
plt.figure(figsize=(12, 8))

# Confusion Matrix
plt.subplot(2, 2, 1)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['≤$100k', '>$100k'], yticklabels=['≤$100k', '>$100k'])
plt.title('Confusion Matrix')

# Feature Importance
plt.subplot(2, 2, 2)
plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
plt.title('Feature Importance')

# Training vs Test Accuracy
plt.subplot(2, 2, 3)
plt.bar(['Training', 'Testing'], [train_accuracy, test_accuracy], 
        color=['lightgreen', 'lightcoral'])
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Decision Tree
plt.subplot(2, 2, 4)
plot_tree(model, max_depth=2, feature_names=feature_names, 
          class_names=['≤$100k', '>$100k'], filled=True, fontsize=8)
plt.title('Decision Tree')

plt.tight_layout()
plt.show()

# Prediction function - FIXED VERSION
def predict_salary(company, job, degree):
    """Predict salary category for given employee characteristics"""
    try:
        company_encoded = le_company.transform([company])[0]
        job_encoded = le_job.transform([job])[0]
        degree_encoded = le_degree.transform([degree])[0]
        
        # Create DataFrame with proper column names to match training data
        input_data = pd.DataFrame({
            'company': [company_encoded],
            'job': [job_encoded],
            'degree': [degree_encoded]
        })
        
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        result = "More than $100k" if prediction[0] == 1 else "$100k or less"
        confidence = max(probability[0]) * 100
        
        print(f"Prediction: {result} (Confidence: {confidence:.1f}%)")
        return prediction[0]
    except ValueError:
        print("Error: Please use valid company, job, and degree values from the dataset")

# Example prediction
if len(df) > 0:
    sample_row = df.iloc[0]
    predict_salary(sample_row['Company'], sample_row['Job'], sample_row['degree'])