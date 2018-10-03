import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


dataset = pd.read_csv('datafiles/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#linkage - algo for hierarchy clustering, ward = minimise variance within each cluster
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidean dist')
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#y_kmeans == [clusterno, column index = 0(x-coord) 1(y-coord)]
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c='red', label='careful')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c='blue', label='standard')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c='green', label='target')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c='orange', label='careless')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c='purple', label='sensible')
plt.title('cluster of client')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()





