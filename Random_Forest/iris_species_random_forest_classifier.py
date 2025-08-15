import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
print(iris.target_names)
print(iris.feature_names)
# print the iris data (top 5 records)
print(iris.data[0:5])
# print the iris labels (0:setosa, 1:versicolor, 2:virginica)
print(iris.target)

data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
print(data)
X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = data['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, Y_train)

#Checking feature importance or finding which feature is most important and contribute in classification
feature_imp = pd.Series(rf_clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)

y_pred = rf_clf.predict(X_test)
print("Accuracy score:", accuracy_score(Y_test, y_pred))

custom_pred = rf_clf.predict([[3, 5, 4, 2]])
print(custom_pred)

#Visualizing data
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
