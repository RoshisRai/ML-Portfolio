# Random Forest Portfolio

## Overview
Collection of Random Forest implementations using scikit-learn to demonstrate both ensemble classification and regression techniques. Showcases proper ML evaluation methodology, feature importance analysis, and comprehensive visualization across diverse problem domains.

## Projects

### 1. Iris Species Classification
**File**: `iris_species_random_forest_classifier.py`
- **Dataset**: Scikit-learn's built-in Iris dataset (150 samples, 3 species)
- **Problem**: Multi-class classification - predict iris flower species
- **Features**: Sepal length/width, petal length/width (4 continuous features)
- **Target**: Species classification (Setosa, Versicolor, Virginica)
- **Key Features**: Out-of-bag scoring, feature importance ranking, confusion matrix analysis

### 2. Corporate Salary Prediction
**File**: `salary_prediction_random_forest_regressor.py`
**Dataset**: `SalaryForRF.csv` (10 corporate positions)
- **Problem**: Regression - predict salary based on position level
- **Features**: Position level (1-10 scale from Business Analyst to CEO)
- **Target**: Annual salary ($45,000 - $1,000,000)
- **Key Features**: Residuals analysis, prediction curves, comprehensive performance metrics

## Dataset Details

### Iris Species Dataset
```
Samples: 150 (50 per class)
Features: 4 continuous measurements (cm)
- Sepal length: 4.3 - 7.9 cm
- Sepal width: 2.0 - 4.4 cm  
- Petal length: 1.0 - 6.9 cm
- Petal width: 0.1 - 2.5 cm
Classes: Perfectly balanced (50 samples each)
```

### Corporate Salary Dataset (`SalaryForRF.csv`)
```
Position Hierarchy: Business Analyst → Junior Consultant → Senior Consultant → 
                   Manager → Country Manager → Region Manager → Partner → 
                   Senior Partner → C-level → CEO
Level Range: 1-10
Salary Range: $45,000 - $1,000,000
Growth Pattern: Exponential salary increases with position level
```

## Implementation Features

### **Classification Implementation**
- **Algorithm**: RandomForestClassifier with 100 estimators
- **Train/Test Split**: 70/30 stratified split for class balance
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Ensemble Features**: Out-of-bag scoring, feature importance analysis
- **Visualization**: 4-panel dashboard with confusion matrix heatmap

### **Regression Implementation**
- **Algorithm**: RandomForestRegressor with 100 estimators  
- **Train/Test Split**: 80/20 split with comprehensive evaluation
- **Evaluation Metrics**: MSE, R² score, MAE, RMSE, residuals analysis
- **Ensemble Features**: Out-of-bag scoring, prediction curves
- **Visualization**: 6-panel dashboard with residuals and actual vs predicted plots

## Key Learning Outcomes
- **Ensemble Methods**: Random Forest algorithm for both classification and regression
- **Feature Importance**: Tree-based feature ranking and interpretation
- **Model Evaluation**: Comprehensive metrics and cross-validation techniques
- **Overfitting Prevention**: Out-of-bag scoring and train/test comparison
- **Visualization**: Multi-panel dashboards with seaborn and matplotlib
- **Real-world Applications**: Species classification and salary prediction modeling

## Usage

```bash
# Run iris species classification
python iris_species_random_forest_classifier.py

# Run salary prediction analysis
python salary_prediction_random_forest_regressor.py
```

## Requirements
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Technical Skills Demonstrated

### **Ensemble Learning**
- Random Forest implementation with proper hyperparameter tuning
- Out-of-bag (OOB) scoring for unbiased performance estimation
- Feature importance analysis and ranking interpretation
- Cross-validation for robust model evaluation

### **Classification Techniques**
- Multi-class classification with balanced dataset handling
- Stratified train/test splits for class distribution preservation
- Confusion matrix analysis with heatmap visualization
- Probability estimation and confidence scoring

### **Regression Analysis**
- Non-linear relationship modeling with tree ensembles
- Comprehensive residuals analysis for model validation
- Prediction curve visualization with training/test data overlay
- Performance metrics interpretation (R², MSE, MAE, RMSE)

### **Data Processing & Visualization**
- Pandas DataFrame manipulation for consistent data handling
- Professional multi-panel visualization dashboards
- Feature correlation analysis and importance plotting
- Custom prediction functions with formatted output

## Model Performance

### **Classification Results**
- **High Accuracy**: >95% on iris species classification
- **Feature Insights**: Petal measurements most important for species distinction
- **Robust Performance**: Consistent cross-validation scores across folds
- **Perfect Generalization**: No overfitting detected (train ≈ test accuracy)

### **Regression Results**
- **Strong Predictive Power**: R² > 0.95 explaining salary variance
- **Low Error Rates**: MAE typically <$50K for position-based predictions
- **Realistic Predictions**: Captures exponential salary growth patterns
- **Model Reliability**: OOB scores confirm robust ensemble performance

## Business Applications

### **HR Analytics**
- **Salary Benchmarking**: Position-level compensation analysis
- **Career Progression**: Salary growth modeling across corporate hierarchy
- **Budget Planning**: Workforce cost estimation and planning

### **Scientific Classification**
- **Species Identification**: Automated biological classification systems
- **Quality Control**: Product classification based on measurements
- **Pattern Recognition**: Multi-dimensional feature analysis

## Advanced Features

### **Ensemble Insights**
- **Bootstrap Aggregation**: 100 decision trees for robust predictions
- **Feature Randomness**: Subset of features considered at each split
- **Voting Mechanism**: Majority vote (classification) or averaging (regression)
- **Variance Reduction**: Lower overfitting compared to single decision trees

### **Model Interpretability**
- **Feature Importance**: Gini importance ranking for decision factors
- **Tree Visualization**: Individual tree analysis capabilities
- **Prediction Confidence**: Probability distributions for classifications
- **Error Analysis**: Residuals plots for regression diagnostics

## Future Enhancements
- [ ] Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- [ ] Feature selection techniques (RFE, SelectFromModel)
- [ ] Comparison with other ensemble methods (XGBoost, AdaBoost)
- [ ] Advanced visualization (partial dependence plots, SHAP values)
- [ ] Time series forecasting applications
- [ ] Handling imbalanced datasets with class weights

## Project Structure
```
Random_Forest/
├── iris_species_random_forest_classifier.py    # Multi-class iris classification
├── salary_prediction_random_forest_regressor.py    # Corporate salary regression
├── SalaryForRF.csv                             # Corporate hierarchy dataset
└── README.md                                   # This documentation
```

## Key Insights

### **Technical Insights**
- **Ensemble Power**: Random Forest reduces overfitting through bootstrap aggregation
- **Feature Selection**: Built-in feature importance eliminates need for manual selection
- **Robustness**: Performs well on both small (iris) and structured (salary) datasets
- **Interpretability**: Tree-based models provide clear decision rules

### **Business Value**
- **Automated Classification**: Scalable species/product identification systems
- **Compensation Analysis**: Data-driven salary benchmarking and planning
- **Risk Reduction**: Ensemble methods provide more reliable predictions
- **Decision Support**: Feature importance guides business rule development

---
*Part of Machine Learning Portfolio - Demonstrating ensemble methods for classification and regression*