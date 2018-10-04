# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
dataset = pd.read_csv('datafiles/Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting the Random Forest Regression to the dataset
#increasing no. of trees will lead to covergence to average rather than more steps
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)


# Predicting a new result
y_pred = regressor.predict(6.5)
print(y_pred)


# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()