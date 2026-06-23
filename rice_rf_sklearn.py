import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
rice = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice.data.features 
y = rice.data.targets 

# variable information 
print(rice.variables) 

le = LabelEncoder()
y_encoded = le.fit_transform(y.values.ravel())

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Accuracy
print("Test Accuracy:",accuracy_score(y_test, y_pred))

#f1_score
print("F1 score:", f1_score(y_test, y_pred))

#recall
print("Sensitivity:", recall_score(y_test, y_pred))

#precision
print("Precision:", precision_score(y_test, y_pred))

#Confusion matrix 
print("Confusion Matrix:")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),index=le.classes_,columns=le.classes_))
