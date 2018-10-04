# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing the dataset
dataset = pd.read_csv('datafiles/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#only need to onehotencode country, gender not necessary
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#avoid dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#make the ANN
classifier = Sequential()

#first hidden layer
#output dim = (input + output) /2 = (11 + 1)/2 = 6 (optimal no of nodes in the hidden layer)
#input dim (no of nodes in input layer) (only at the first hidden layer)
#relu = rectifier activation
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu', input_dim=11))

#second hidden layer
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu'))

#output layer (only 1 output since binary outcome)
classifier.add(Dense(output_dim=1, kernel_initializer='uniform', activation='sigmoid'))

#compile ann
#stochastic gradient descent algo = adam
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
