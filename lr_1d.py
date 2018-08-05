import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))
X = np.array(X)
Y = np.array(Y)

# apply eqn to calc a and b
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

Yhat = a * X + b


#calculate r-squared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)



plt.scatter(X, Y)
plt.title("r-squared: " + str(r2))
plt.plot(X, Yhat)

plt.show()





