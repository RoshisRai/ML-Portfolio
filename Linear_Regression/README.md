# Linear Regression Portfolio

## Overview
Collection of Linear Regression implementations using scikit-learn to demonstrate fundamental machine learning concepts across diverse datasets and problem domains.

## Projects

### 1. California Housing Price Prediction
**File**: `california_housing_regression.py`
- **Dataset**: California Housing (20,640 samples, 8 features)
- **Problem**: Predict median house values based on demographic and geographic data
- **Features**: Median income, house age, average rooms, population, latitude/longitude
- **Results**: MSE ~0.556 (~$74,600 average error)

### 2. Diabetes Progression Prediction
**File**: `diabetes_regression.py`
- **Dataset**: Diabetes Dataset (442 samples, 10 features)
- **Problem**: Predict disease progression based on patient physiological measurements
- **Features**: Age, sex, BMI, blood pressure, serum measurements
- **Includes**: Comprehensive visualization, residual analysis, feature importance ranking

### 3. Transportation Cost Prediction
**File**: `transportation_price_regression.py`
- **Dataset**: Custom transportation data (15 samples)
- **Problem**: Predict travel costs based on distance
- **Features**: Distance in kilometers
- **Focus**: Simple univariate regression demonstration

## Implementation Features
- **Algorithm**: Linear Regression (Ordinary Least Squares)
- **Train/Test Split**: Proper data splitting with random states for reproducibility
- **Evaluation Metrics**: MSE, R² score, RMSE (where applicable)
- **Visualization**: Scatter plots, regression lines, residual analysis
- **Documentation**: Comprehensive code comments and docstrings

## Common Learning Outcomes
- Data loading and preprocessing techniques
- Train/test split methodology
- Linear regression model training and evaluation
- Performance metrics interpretation
- Visualization of regression results
- Understanding different problem domains (real estate, healthcare, transportation)

## Usage
```bash
# Run individual projects
python california_housing_regression.py
python diabetes_regression.py
python transportation_price_regression.py
```

## Requirements
```
scikit-learn>=1.0.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0  # For diabetes regression feature analysis
```

## Technical Skills Demonstrated
- **Data Handling**: Multiple dataset types and sizes
- **Model Training**: Scikit-learn LinearRegression implementation
- **Evaluation**: MSE, R², residual analysis
- **Visualization**: Matplotlib plotting and analysis
- **Code Quality**: Professional documentation and structure

## Future Enhancements
- [ ] Add feature scaling implementations
- [ ] Include cross-validation analysis
- [ ] Compare with polynomial regression
- [ ] Add more diverse datasets
- [ ] Implement regularized regression (Ridge, Lasso)
- [ ] Performance optimization techniques

## Project Structure
```
Linear_Regression/
├── california_housing_regression.py    # Real estate price prediction
├── diabetes_regression.py              # Healthcare progression modeling
├── transportation_price_regression.py  # Cost estimation model
└── README.md                           # This documentation
```

---
*Part of Machine Learning Portfolio - Demonstrating fundamental regression techniques across multiple domains*