import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('datafiles/Mall_Customers.csv')
#annual income and spending score
X = dataset.iloc[:,[3,4]].values
wcss = []
#kmeans++ will avoid the random initialisaiton problem
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    #compute the within sums of squares kemans.inertia
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('no of cluster')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

#y_kmeans == [clusterno, column index = 0(x-coord) 1(y-coord)]
plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red', label='careful')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue', label='standard')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green', label='target')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c='orange', label='careless')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=100, c='purple', label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow', label='Centroids')
plt.title('cluster of client')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()

