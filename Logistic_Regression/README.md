# Logistic Regression Portfolio

## Overview
Collection of Logistic Regression implementations using scikit-learn to demonstrate binary classification techniques across diverse problem domains.

## Projects

### 1. Student Performance Classification
**File**: `student_pass_fail_classification.py`
- **Dataset**: Synthetic student performance data (30 samples, 7 features)
- **Problem**: Binary classification - predict student pass/fail outcomes
- **Features**: Study hours, attendance %, previous GPA, assignment completion, participation score, family support, extracurricular activities
- **Target**: Pass_Or_Fail (binary: 0=Fail, 1=Pass)
- **Includes**: Confusion matrix, feature importance, probability distribution

## Implementation Features
- **Algorithm**: Logistic Regression with L2 regularization
- **Data Handling**: Synthetic data generation with realistic student performance patterns
- **Train/Test Split**: Stratified 75/25 split for class balance
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Visualization**: Three-panel analysis (confusion matrix, feature importance, probabilities)

## Key Learning Outcomes
- Binary classification workflow
- Model training and evaluation
- Classification metrics interpretation
- Feature importance through coefficients
- Confusion matrix analysis
- Prediction probability visualization

## Usage
```bash
python student_pass_fail_classification.py
```

## Requirements
```
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Technical Skills Demonstrated
- **Data Processing**: Pandas DataFrame creation and exploration
- **Classification**: Scikit-learn LogisticRegression implementation
- **Evaluation**: Classification metrics and confusion matrix
- **Visualization**: Multi-panel plots with matplotlib/seaborn

## Future Enhancements
- [ ] Add ROC curve and AUC analysis
- [ ] Implement cross-validation
- [ ] Feature scaling and normalization
- [ ] Hyperparameter tuning

---
*Part of Machine Learning Portfolio - Demonstrating classification techniques*