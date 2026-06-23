import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#Loading the CSV
df = pd.read_csv('Iris.csv', sep=';', header=None,names=['feature1', 'feature2', 'feature3','feature4','Type of Iris'])
df.head()
X = df[['feature1', 'feature2','feature3','feature4']]
y = df['Type of Iris']


#Maxmin normalisation
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#One-hot encoding
encoding = OneHotEncoder()
y_encoded = encoding.fit_transform(y.values.reshape(-1, 1)).toarray()

# for splitting data to train and testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, warm_start=True, random_state=42)

#Train the network and printing loss after every epoch
epochs = 500
unique_classes = [0, 1, 2]

for epoch in range(1, epochs + 1):
    # Train for one epoch
    model.partial_fit(X_train, y_train, classes=unique_classes)
    
    # Get probabilities for training and validation sets
    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    
    # Calculate sum-of-squares loss
    train_loss = np.sum((train_probs - y_train) ** 2)
    val_loss = np.sum((val_probs - y_val) ** 2)
    
    # Print loss after every epoch
    print("Epoch", epoch, "Train Loss:", round(train_loss, 3), "Val Loss:", round(val_loss, 3))

#Accuracy
y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("Test Accuracy:", accuracy)
