# Linear Regression - California Housing Dataset

## Overview
Implementation of Linear Regression algorithm using scikit-learn to predict California housing prices based on demographic and geographic features.

## Dataset
- **Source**: California Housing Dataset (scikit-learn)
- **Samples**: 20,640 houses
- **Features**: 8 (median income, house age, average rooms, etc.)
- **Target**: Median house value (in hundreds of thousands of dollars)

## Features
1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

## Implementation Details
- **Algorithm**: Linear Regression (Ordinary Least Squares)
- **Train/Test Split**: 80/20 (16,512/4,128 samples)
- **Random State**: 42 (for reproducibility)
- **Evaluation Metric**: Mean Squared Error (MSE)

## Results
- **MSE**: ~0.556
- **Interpretation**: Average prediction error of approximately $74,600

## Usage
```bash
python LinearRegression.py
```

## Requirements
```
scikit-learn>=1.0.0
numpy>=1.21.0
```

## Key Learning Outcomes
- Data loading and preprocessing
- Train/test split methodology
- Linear regression model training
- Model evaluation using MSE
- Understanding of regression problem setup

## Future Improvements
- Add R² score evaluation
- Implement feature scaling
- Add data visualization
- Compare with polynomial regression
- Cross-validation implementation