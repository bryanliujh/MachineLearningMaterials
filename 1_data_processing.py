import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('datafiles/Data.csv')
#take all the columns except the last one (:-1)
X = dataset.iloc[:, :-1].values
#takes the last column
y = dataset.iloc[:, 3].values

#takes care of missing data
#axis = 0 (mean of col) axis = 1(mean of rows)
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#takes col from matrix X index 1 to 2 and fit an imputer on it
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#encoding categorical data, label encoder = ordinal, one hot encoding = non-ordinal
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
#a better encoder that create n columns based on n categories, instead of assign one column to all category as it might imply france > germany
onehotEncoder = OneHotEncoder(categorical_features=[0])
X = onehotEncoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
#encode dependent variable purchased can use label encoder as it is dependent variable and will be recognised as a category with no order
y = labelEncoder_y.fit_transform(y)

#splitting dataset into test set and training set, set test data to 20% of train set
#random state is to ensure that it will generate the same set everytime (not necessary)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)


#feature scaling (normalise data to be on the same scale, so no variable will dominate the others)
# no need scale y as it is a categorical variable (0,1)
#We use fit_transform() on the train data so that we learn the parameters of scaling on the train data and in the same time we scale the train data.
# We only use transform() on the test data because we use the scaling paramaters learned on the train data to scale the test data.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


print(X_train)
