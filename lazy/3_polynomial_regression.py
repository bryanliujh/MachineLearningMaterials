import numpy as np
import matplotlib.pyplot as plt

#load the data
X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1,x,x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1],Y)
plt.show()

#calc weights
w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat = np.dot(X,w)


#compute rquared
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1)/d2.dot(d2)



# to plot our quadratic model predictions, let's
# create a line of x's and calculate the predicted y's
x_line = np.linspace(X[:,1].min(), X[:,1].max())
y_line = w[0] + w[1] * x_line + w[2] * x_line * x_line


#plot all together
plt.scatter(X[:,1],Y)
plt.title("r-squared: " + str(r2))
plt.plot(x_line,y_line)
plt.show()

