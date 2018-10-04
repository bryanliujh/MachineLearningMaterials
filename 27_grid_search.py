# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC



# Importing the dataset
dataset = pd.read_csv('datafiles/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

#applying k-fold cross validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

#take the mean and std of 10 accuracies
print(accuracies.mean())
#if std = 6% and mean = 90%, means model accuracy will be about 84% to 96%
print(accuracies.std())


#Applying Grid Search to find the best model and best parameters
parameters = [{'C':[1,10,100,1000], 'kernel':['linear']},
              {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}]

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

print(best_accuracy)

best_parameters = grid_search.best_params_

print(best_parameters)
