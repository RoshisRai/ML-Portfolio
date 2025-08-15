# Decision Trees Portfolio

## Overview
Collection of Decision Tree implementations using scikit-learn to demonstrate both regression and classification techniques across diverse problem domains. Showcases proper ML evaluation methodology with train/test splits, overfitting prevention, and comprehensive visualization.

## Projects

### 1. Game Development Profit Prediction (Regression)
**File**: `game_profit_decision_tree_regression.py`
- **Dataset**: Synthetic game development data (14 game types)
- **Problem**: Predict expected profit based on production costs
- **Features**: Production cost (single feature regression)
- **Target**: Expected profit in dollars
- **Key Features**: 4-panel dashboard with training/test evaluation, cross-validation analysis

### 2. Employee Salary Classification
**File**: `salary_prediction_decision_tree_classifier.py`
**Dataset**: `Salarydata.csv` (16 employee records)
- **Problem**: Binary classification - predict if salary exceeds $100k
- **Features**: Company (Google, Facebook, ABC Pharma), Job Title, Education Level
- **Target**: Salary_more_than_100k (binary: 0=≤$100k, 1=>$100k)
- **Key Features**: Comprehensive evaluation with confusion matrix and decision tree visualization

## Implementation Features

### **Regression Implementation**
- **Algorithm**: DecisionTreeRegressor with overfitting prevention
- **Train/Test Split**: 70/30 split with proper evaluation
- **Evaluation Metrics**: MSE, R² score, cross-validation (3-fold)
- **Overfitting Prevention**: max_depth=4, min_samples_split=3, min_samples_leaf=2
- **Visualization**: Training/test comparison, performance metrics, prediction curves

### **Classification Implementation**
- **Algorithm**: DecisionTreeClassifier with complexity controls
- **Train/Test Split**: 80/20 stratified split for class balance
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Overfitting Prevention**: max_depth=5, min_samples_split=5, min_samples_leaf=2
- **Visualization**: Confusion matrix heatmap, feature importance, decision tree plot

## Dataset Details

### Game Development Data
```
Game Types: Asset Flip, Text Based, Visual Novel, 2D Pixel Art, Strategy, 
           FPS, Simulator, Racing, RPG, Sandbox, Open-World, MMOFPS, MMORPG
Cost Range: $100 - $30,000
Profit Range: $1,000 - $80,000
```

### Employee Salary Data (`Salarydata.csv`)
```
Companies: Google, Facebook, ABC Pharma
Jobs: Sales Executive, Business Manager, Computer Programmer  
Education: Bachelors, Masters
Target: Binary salary classification (≤$100k vs >$100k)
```

## Key Learning Outcomes
- **Decision Tree Algorithms**: Both regression and classification implementations
- **Overfitting Prevention**: Proper hyperparameter tuning and complexity control
- **Model Evaluation**: Train/test methodology, cross-validation, performance metrics
- **Data Handling**: CSV loading, categorical encoding with LabelEncoder
- **Visualization**: Multi-panel dashboards with matplotlib and seaborn
- **Real-world Applications**: Game development economics and HR salary analysis

## Usage

```bash
# Run regression analysis
python game_profit_decision_tree_regression.py

# Run classification analysis  
python salary_prediction_decision_tree_classifier.py
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

### **Model Development**
- Decision tree hyperparameter tuning for overfitting prevention
- Proper train/test split methodology with stratification
- Cross-validation for robust model evaluation
- Feature importance analysis and interpretation

### **Data Processing**
- Synthetic data generation for regression problems
- CSV data loading and preprocessing
- Categorical variable encoding with LabelEncoder
- Proper DataFrame handling to avoid sklearn warnings

### **Evaluation & Visualization**
- Comprehensive performance metrics (MSE, R², accuracy, precision, recall)
- Multi-panel visualization dashboards
- Confusion matrix analysis with seaborn heatmaps
- Decision tree visualization with plot_tree
- Training vs testing performance comparison

## Model Performance

### **Regression Model**
- **Overfitting Prevention**: R² difference (train-test) < 0.2
- **Cross-Validation**: Consistent performance across folds
- **Predictions**: Production cost → profit estimation

### **Classification Model**
- **Balanced Evaluation**: Stratified sampling maintains class distribution
- **Feature Analysis**: Company and education level as key predictors
- **Interpretability**: Visual decision tree shows decision rules

## Future Enhancements
- [ ] Random Forest comparison for ensemble methods
- [ ] Feature scaling and advanced preprocessing
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Multi-class classification examples
- [ ] Feature selection techniques
- [ ] Advanced visualization with decision boundaries

## Project Structure
```
Decision_Trees/
├── game_profit_decision_tree_regression.py    # Game development profit prediction
├── salary_prediction_decision_tree_classifier.py    # Employee salary classification
├── Salarydata.csv                            # Employee dataset
└── README.md                                 # This documentation
```

## Key Insights

### **Business Applications**
- **Game Development**: Cost-profit relationship modeling for budget planning
- **HR Analytics**: Salary prediction based on role and qualifications
- **Decision Making**: Interpretable models for business rule extraction

### **Technical Insights**
- **Overfitting Control**: Essential for small datasets and tree-based models
- **Feature Importance**: Decision trees provide natural feature ranking
- **Interpretability**: Tree visualization enables business rule understanding

---
*Part of Machine Learning Portfolio - Demonstrating tree-based algorithms for regression and classification*