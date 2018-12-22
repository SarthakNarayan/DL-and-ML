import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import numpy as np

# dataset Iris
from sklearn import datasets

iris = datasets.load_iris()
# We only take two features so that we can show clustering in matplotlib
x = iris.data[:,:2] 
y = iris.target

# original data without clustering
plt.figure(figsize=(12,8))
plt.scatter(x[:,0], x[:,1])
plt.xlabel('Sepia Length')
plt.ylabel('Sepia Width')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
assignments = kmeans.labels_
clusters = kmeans.cluster_centers_
#print(assignments)
#print(clusters)

def ScatterPlot(X , Y, assignments, centers):
  plt.figure(figsize=(12,8))
  cmap = ListedColormap(['red', 'green', 'blue'])
  plt.scatter(X, Y, c=assignments, cmap=cmap)
  plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)), 
                marker='+', s=400, cmap=cmap)
  plt.xlabel('Sepia Length')
  plt.ylabel('Sepia Width')
  plt.show()
  
ScatterPlot(x[:,0], x[:,1], assignments, clusters)
score = kmeans.score(x)
print(np.abs(score))

K = [2, 3, 4, 5, 6]
SSe = [57.98, 37.12, 27.98, 20.97, 17.23]
plt.figure(figsize=(12,8))
plt.plot(K,SSe)
plt.xlabel('Number of Clusters')
plt.ylabel('score')
plt.show()
