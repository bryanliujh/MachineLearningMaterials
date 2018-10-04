import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('datafiles/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3, random_state=0)

#fitting the regression model to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting test set result
y_pred = regressor.predict(X_test)

#real salary
print(y_test)
#predicted salary
print(y_pred)

#visualise the training set result (real)
plt.scatter(X_train, y_train, color = 'red')
#regression line (predicted)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


#compare training model against test set to see if it is a good model
#visualise the test set result (real)
plt.scatter(X_test, y_test, color = 'red')
#no need change this to X_test as regressor is trained (fitted) into training set
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


