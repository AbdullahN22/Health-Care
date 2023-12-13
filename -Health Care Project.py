import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_excel(r'C:\Users\najeh\OneDrive\Desktop\Project\Health Care\data.xlsx')

# Data Exploration
print("Data Shape:", data.shape)
print("Missing Values:\n", data.isnull().sum())
print("Unique Values per Column:\n", data.nunique(axis=0))
print("Target Value Counts:\n", data['target'].value_counts())
print("Data Summary Statistics:\n", data.describe())

# Exploratory Data Analysis
corr = data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title("Correlation Heatmap", size=25)

subdata = data[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]
sns.pairplot(subdata)

sns.catplot(x='target', y='oldpeak', hue='slope', kind='bar', data=data)
plt.title('ST Depression vs. Heart Disease', size=25)
plt.xlabel('Heart Disease', size=20)
plt.ylabel('ST Depression', size=20)

plt.figure(figsize=(12, 8))
sns.boxplot(x='target', y='thalach', hue="sex", data=data)
plt.title("ST Depression Level vs. Heart Disease", fontsize=20)
plt.xlabel("Heart Disease Target", fontsize=16)
plt.ylabel("ST Depression induced by exercise relative to rest", fontsize=16)

# Machine Learning + Predictive Analytics
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)

# Normalizing for x
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Logistic Regression
LR = LogisticRegression()
LR.fit(x_train, y_train)
y_pred1 = LR.predict(x_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred1))
CM1 = confusion_matrix(y_test, y_pred1)
AS1 = accuracy_score(y_test, y_pred1)
print('Confusion Matrix:\n', CM1)
print('Accuracy Score:', AS1)

# Random Forest
RFC = RandomForestClassifier()
RFC.fit(x_train, y_train)
y_pred2 = RFC.predict(x_test)
print("Random Forest:")
print(classification_report(y_test, y_pred2))
CM2 = confusion_matrix(y_test, y_pred2)
AS2 = accuracy_score(y_test, y_pred2)
print('Confusion Matrix:\n', CM2)
print('Accuracy Score:', AS2)

# Feature Importance
feature_importance = RFC.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f'Feature {i}: Importance Score = {importance:.5f}')

# Plot the top features
index = data.columns[:-1]
feature_importance_series = pd.Series(feature_importance, index=index)
top_features = feature_importance_series.nlargest(13)
top_features.plot(kind='barh', colormap='winter')

# Compare predicted and actual values
y_test_array = np.array(y_test)
y_pred = RFC.predict(x_test)
comparison = np.column_stack((y_pred.reshape(len(y_pred), 1), y_test_array.reshape(len(y_test_array), 1)))
print("Predicted vs. Actual:\n", comparison)
