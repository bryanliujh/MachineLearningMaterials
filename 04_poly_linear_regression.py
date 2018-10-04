import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('datafiles/Position_Salaries.csv')
# [:,1:2] will make X a matrix instead of [:,1] which makes it a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#no need split into train and test as too little data

#fit linear regression
linear_regressor = LinearRegression()
linear_regressor.fit(X,y)

#fit polynomial linear regression
#poly regressor to transform values into polynomial
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)


#compare linear and polynomial models

#visualise linear
plt.scatter(X, y, color='red')
plt.plot(X, linear_regressor.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualise poly
#arrange and reshape to smooth the curve (every increment by 0.1)
X_grid = np.arange(min(X), max(X), 0.1)
#reshape into matrix
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='red')
#plot(X,y)
plt.plot(X_grid, linear_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color='blue')
plt.title('Poly Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#predict level 6.5 with linear regression
print(linear_regressor.predict(6.5))

#predict level 6.5 with poly linear regression
print(linear_regressor_2.predict(poly_regressor.fit_transform(6.5)))

