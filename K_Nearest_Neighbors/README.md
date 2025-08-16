# K-Nearest Neighbors (KNN) Portfolio

## Overview
Collection of K-Nearest Neighbors implementations using scikit-learn to demonstrate both distance-based regression and classification techniques. Showcases proper feature scaling, hyperparameter tuning, and comprehensive evaluation methodology essential for distance-based algorithms.

## Projects

### 1. Game Development Profit Prediction (Regression)
**File**: `game_profit_knn_regression.py`
- **Dataset**: Synthetic game development data (14 game types)
- **Problem**: Predict expected profit based on production costs
- **Features**: Production cost (single feature regression)
- **Target**: Expected profit in dollars ($1,000 - $80,000)
- **Key Features**: Cross-validation K selection, feature scaling, 4-panel visualization

### 2. Iris Species Classification
**File**: `iris_species_knn_classification.py`
- **Dataset**: Scikit-learn's built-in Iris dataset (150 samples, 3 species)
- **Problem**: Multi-class classification - predict iris flower species
- **Features**: Sepal length/width, petal length/width (4 continuous features)
- **Target**: Species classification (Setosa, Versicolor, Virginica)
- **Key Features**: Train/test accuracy comparison for K selection, confusion matrix analysis

## Dataset Details

### Game Development Data
```
Game Types: Asset Flip, Text Based, Visual Novel, 2D Pixel Art, Strategy, 
           FPS, Simulator, Racing, RPG, Sandbox, Open-World, MMOFPS, MMORPG
Cost Range: $100 - $30,000
Profit Range: $1,000 - $80,000
Pattern: Non-linear relationship between production cost and profit
```

### Iris Species Dataset
```
Samples: 150 (50 per class - perfectly balanced)
Features: 4 continuous measurements (cm)
- Sepal length: 4.3 - 7.9 cm
- Sepal width: 2.0 - 4.4 cm  
- Petal length: 1.0 - 6.9 cm
- Petal width: 0.1 - 2.5 cm
Classes: Setosa, Versicolor, Virginica
```

## Implementation Features

### **Regression Implementation**
- **Algorithm**: KNeighborsRegressor with optimal K selection (K=1-7)
- **Feature Scaling**: StandardScaler (essential for distance-based algorithms)
- **K Selection**: Cross-validation with R² scoring for robust selection
- **Evaluation**: Train/test R² comparison, overfitting detection
- **Visualization**: Prediction curves, K selection plot, performance comparison

### **Classification Implementation**
- **Algorithm**: KNeighborsClassifier with optimal K selection (K=1-15)
- **Feature Scaling**: StandardScaler applied to 4-dimensional feature space
- **K Selection**: Train/test accuracy comparison for optimal performance
- **Evaluation**: Classification report, confusion matrix, cross-validation
- **Visualization**: K selection curves, confusion matrix heatmap, accuracy comparison

## Key Learning Outcomes
- **Distance-Based Learning**: Understanding how KNN uses proximity for predictions
- **Feature Scaling**: Critical importance of standardization for distance calculations
- **Hyperparameter Tuning**: Systematic K selection via cross-validation
- **Curse of Dimensionality**: Performance considerations in multi-dimensional spaces
- **Local Pattern Recognition**: KNN's strength in capturing local data relationships
- **Lazy Learning**: No explicit training phase - all computation during prediction

## Usage

```bash
# Run game development profit prediction
python game_profit_knn_regression.py

# Run iris species classification
python iris_species_knn_classification.py
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

### **Distance-Based Learning**
- Euclidean distance calculations in multi-dimensional space
- Feature scaling with StandardScaler for consistent distance metrics
- K-value selection through systematic hyperparameter tuning
- Understanding of local vs global learning approaches

### **Hyperparameter Optimization**
- Cross-validation for robust K selection in regression
- Train/test accuracy comparison for optimal K in classification
- Bias-variance tradeoff analysis (low K = high variance, high K = high bias)
- Overfitting detection through performance gap analysis

### **Comprehensive Evaluation**
- Regression metrics: R² score, train/test comparison, prediction curves
- Classification metrics: Accuracy, precision, recall, confusion matrix
- Cross-validation for robust performance estimation
- Visual analysis of K selection and model performance

### **Data Processing & Visualization**
- Feature standardization for distance-based algorithms
- Multi-panel dashboard visualization (4 panels per project)
- Custom prediction functions with formatted output
- Professional data handling with pandas DataFrames

## Model Performance

### **Regression Results**
- **Strong Performance**: High R² scores with optimal K selection
- **Local Pattern Capture**: Effective modeling of non-linear cost-profit relationships
- **Robust K Selection**: Cross-validation ensures generalizable performance
- **Feature Scaling Impact**: Demonstrates necessity of standardization

### **Classification Results**
- **High Accuracy**: >95% accuracy on iris species classification
- **Perfect Separation**: Clear decision boundaries between species
- **Optimal K Detection**: Systematic selection of best neighborhood size
- **Balanced Performance**: Consistent accuracy across all three species

## Business Applications

### **Game Development Economics**
- **Budget Planning**: Production cost vs profit relationship modeling
- **Investment Decisions**: ROI estimation for different game types
- **Market Analysis**: Genre-specific profitability patterns
- **Resource Allocation**: Optimal budget distribution strategies

### **Scientific Classification**
- **Species Identification**: Automated biological classification systems
- **Quality Control**: Product classification based on measurements
- **Medical Diagnosis**: Pattern recognition in clinical data
- **Image Recognition**: Feature-based classification systems

## Advanced KNN Concepts

### **Algorithm Characteristics**
- **Lazy Learning**: No training phase - all computation at prediction time
- **Instance-Based**: Stores all training instances for comparison
- **Non-Parametric**: No assumptions about underlying data distribution
- **Local Learning**: Predictions based on local neighborhood patterns

### **Distance Metrics**
- **Euclidean Distance**: Default metric for continuous features
- **Manhattan Distance**: Alternative for high-dimensional spaces
- **Minkowski Distance**: Generalization of Euclidean and Manhattan
- **Feature Scaling**: Essential for consistent distance calculations

### **Hyperparameter Considerations**
- **K Value Selection**: Balance between bias (high K) and variance (low K)
- **Odd vs Even K**: Odd values prevent ties in classification
- **Dataset Size Impact**: Larger datasets allow for higher K values
- **Curse of Dimensionality**: Performance degradation in high dimensions

## Future Enhancements
- [ ] Distance metric comparison (Euclidean vs Manhattan vs Minkowski)
- [ ] Weighted KNN implementation (distance-based weighting)
- [ ] High-dimensional datasets and curse of dimensionality analysis
- [ ] KD-Tree and Ball-Tree implementations for efficiency
- [ ] Real-world datasets with missing values and outliers
- [ ] Comparison with other instance-based learning algorithms

## Project Structure
```
K_Nearest_Neighbors/
├── game_profit_knn_regression.py          # Game development profit prediction
├── iris_species_knn_classification.py     # Iris species classification
└── README.md                              # This documentation
```

## Key Insights

### **Technical Insights**
- **Feature Scaling is Critical**: Distance-based algorithms require standardized features
- **K Selection is Art and Science**: Balance between overfitting and underfitting
- **Local Learning Power**: KNN excels at capturing local patterns and irregularities
- **Computational Trade-offs**: Simple algorithm but expensive prediction phase

### **Business Value**
- **No Distributional Assumptions**: Works well with complex, non-linear relationships
- **Interpretable Decisions**: Can examine nearest neighbors for prediction rationale
- **Flexible Applications**: Effective for both regression and classification problems
- **Baseline Performance**: Excellent algorithm for initial model comparison

### **When to Use KNN**
- **Small to Medium Datasets**: Performance degrades with very large datasets
- **Non-Linear Relationships**: Captures complex patterns without explicit modeling
- **Local Pattern Importance**: When nearby observations are most relevant
- **Baseline Modeling**: Quick implementation for initial performance benchmarks

---
*Part of Machine Learning Portfolio - Demonstrating distance-based learning algorithms for regression and classification*