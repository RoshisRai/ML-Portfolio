"""
Gaussian Naive Bayes Classification for Tennis Playing Prediction
================================================================
This script demonstrates Gaussian Naive Bayes classification to predict whether
tennis will be played based on weather conditions. Includes feature encoding,
comprehensive evaluation, and decision-making interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Tennis playing dataset
data = {
    'weather': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
    'temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 
                    'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'play_tennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 
                    'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Create DataFrame
df = pd.DataFrame(data)
print("Tennis Playing Dataset:")
print(df.to_string(index=False))

print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['play_tennis'].value_counts())

print(f"\nFeature distributions:")
print(f"Weather: {df['weather'].value_counts().to_dict()}")
print(f"Temperature: {df['temperature'].value_counts().to_dict()}")

# Encode categorical features
le_weather = LabelEncoder()
le_temperature = LabelEncoder()
le_play = LabelEncoder()

# Fit and transform features
weather_encoded = le_weather.fit_transform(df['weather'])
temp_encoded = le_temperature.fit_transform(df['temperature'])
play_encoded = le_play.fit_transform(df['play_tennis'])

print(f"\nEncoding mappings:")
print(f"Weather: {dict(zip(le_weather.classes_, le_weather.transform(le_weather.classes_)))}")
print(f"Temperature: {dict(zip(le_temperature.classes_, le_temperature.transform(le_temperature.classes_)))}")
print(f"Play Tennis: {dict(zip(le_play.classes_, le_play.transform(le_play.classes_)))}")

# Prepare features and target
X = pd.DataFrame({
    'weather_encoded': weather_encoded,
    'temperature_encoded': temp_encoded
})
y = play_encoded

print(f"\nEncoded dataset:")
encoded_df = pd.DataFrame({
    'weather': df['weather'],
    'weather_encoded': weather_encoded,
    'temperature': df['temperature'],
    'temp_encoded': temp_encoded,
    'play_tennis': df['play_tennis'],
    'play_encoded': play_encoded
})
print(encoded_df.to_string(index=False))

# Since dataset is small, use stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nData split: Training={len(X_train)}, Testing={len(X_test)}")

# Train Gaussian Naive Bayes
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

# Evaluate model
y_pred_test = gnb_classifier.predict(X_test)
train_accuracy = gnb_classifier.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nModel Performance:")
print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Testing Accuracy: {test_accuracy:.3f}")

# Cross-validation (with small dataset, use smaller cv)
cv_scores = cross_val_score(gnb_classifier, X, y, cv=3)  # 3-fold due to small dataset
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=le_play.classes_))

# Prediction function with interpretable input/output
def predict_tennis_playing(weather_condition, temperature_condition):
    """Predict tennis playing based on weather and temperature"""
    try:
        # Encode input conditions
        weather_code = le_weather.transform([weather_condition])[0]
        temp_code = le_temperature.transform([temperature_condition])[0]
        
        # Make prediction
        input_data = pd.DataFrame({
            'weather_encoded': [weather_code],
            'temperature_encoded': [temp_code]
        })
        
        prediction = gnb_classifier.predict(input_data)
        probabilities = gnb_classifier.predict_proba(input_data)
        
        # Decode prediction
        result = le_play.inverse_transform(prediction)[0]
        confidence = max(probabilities[0]) * 100
        
        print(f"Conditions: {weather_condition} weather, {temperature_condition} temperature")
        print(f"Prediction: {result} (Confidence: {confidence:.1f}%)")
        print(f"Probabilities: No={probabilities[0][0]:.3f}, Yes={probabilities[0][1]:.3f}")
        
        return result
    
    except ValueError as e:
        print(f"Error: Invalid input. Please use valid weather/temperature conditions.")
        print(f"Valid weather: {list(le_weather.classes_)}")
        print(f"Valid temperature: {list(le_temperature.classes_)}")
        return None

# Visualization
plt.figure(figsize=(15, 10))

# Dataset overview
plt.subplot(2, 3, 1)
play_counts = df['play_tennis'].value_counts()
colors = ['lightcoral', 'lightgreen']
plt.pie(play_counts.values, labels=play_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Tennis Playing Distribution')

# Weather vs Play Tennis
plt.subplot(2, 3, 2)
weather_play_crosstab = pd.crosstab(df['weather'], df['play_tennis'])
weather_play_crosstab.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('Weather vs Tennis Playing')
plt.xlabel('Weather Condition')
plt.ylabel('Count')
plt.legend(title='Play Tennis')
plt.xticks(rotation=45)

# Temperature vs Play Tennis
plt.subplot(2, 3, 3)
temp_play_crosstab = pd.crosstab(df['temperature'], df['play_tennis'])
temp_play_crosstab.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.title('Temperature vs Tennis Playing')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.legend(title='Play Tennis')
plt.xticks(rotation=45)

# Feature correlation heatmap
plt.subplot(2, 3, 4)
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Matrix')

# Confusion Matrix (if test set has samples)
plt.subplot(2, 3, 5)
if len(y_test) > 0:
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_play.classes_, yticklabels=le_play.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
else:
    plt.text(0.5, 0.5, 'Small dataset\nLimited test samples', 
             ha='center', va='center', fontsize=12)
    plt.title('Test Set Too Small')
    plt.axis('off')

# Cross-validation scores
plt.subplot(2, 3, 6)
plt.bar(range(len(cv_scores)), cv_scores, color='gold')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
            label=f'Mean: {cv_scores.mean():.3f}')
plt.title('Cross-Validation Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Example predictions for all possible combinations
print(f"\nAll Possible Predictions:")
print("=" * 60)
weather_conditions = le_weather.classes_
temperature_conditions = le_temperature.classes_

for weather in weather_conditions:
    for temp in temperature_conditions:
        predict_tennis_playing(weather, temp)
        print("-" * 40)

# Model insights
print(f"\nModel Insights:")
print(f"• Gaussian Naive Bayes assumes features follow normal distribution")
print(f"• Small dataset ({len(df)} samples) - results may vary with more data")
print(f"• Feature independence assumed (weather and temperature treated as independent)")
print(f"• Model learns probability distributions for each class and feature")

# Feature statistics by class
print(f"\nFeature Statistics by Class:")
for class_idx, class_name in enumerate(le_play.classes_):
    class_mask = (y == class_idx)
    if np.any(class_mask):
        weather_mean = X.loc[class_mask, 'weather_encoded'].mean()
        temp_mean = X.loc[class_mask, 'temperature_encoded'].mean()
        print(f"{class_name}: Weather avg={weather_mean:.2f}, Temperature avg={temp_mean:.2f}")