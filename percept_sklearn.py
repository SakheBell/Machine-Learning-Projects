import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, classification_report

#Loading the CSV
df = pd.read_csv('Percept1.csv', sep=';', header=None, names=['feature1', 'feature2', 'target'])
df.head()

X = df[['feature1', 'feature2']]
y = df['target']

# for splitting data to train and testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
model = Perceptron()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#visualization
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))

