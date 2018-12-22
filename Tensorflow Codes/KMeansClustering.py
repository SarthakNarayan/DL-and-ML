import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# dataset Iris
from sklearn import datasets

iris = datasets.load_iris()
# We only take two features so that we can show clustering in matplotlib
x = iris.data[:,:2] 
y = iris.target

# original data without clustering
plt.scatter(x[:,0], x[:,1])
plt.xlabel('Sepia Length')
plt.ylabel('Sepia Width')

# According to TensorFlow docs, for K means clustering Estimator, we need to 
# provide input_fn() to fit()
def input_fn():
    return tf.constant(np.array(x), tf.float32, x.shape),None

'''
here, we already know the number of classes as 3, so we set num_clusters=3. 
Normally, we are not aware of what the number of clusters should be; 
in that case, a common methodology is the elbow method:
'''
kmeans = tf.contrib.learn.KMeansClustering(num_clusters=3, relative_tolerance=0.0001, random_seed=2)
kmeans.fit(input_fn=input_fn)

'''
We find the clusters using the clusters() method and, to each input point,
we assign the cluster index using the predict_cluster_idx() method:
'''
clusters = kmeans.clusters()
# gives an array of centre of the clusters
print(clusters)
# assignments gives the indexes of feature points i.e. for each feature row returns the index 
# of the cluster to which it is allocated
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))
print(assignments)

def ScatterPlot(X , Y, assignments, centers):
  plt.figure(figsize=(12,8))
  cmap = ListedColormap(['red', 'green', 'blue'])
  plt.scatter(X, Y, c=assignments, cmap=cmap)
  plt.xlabel('Sepia Length')
  plt.ylabel('Sepia Width')
  plt.show()
  
ScatterPlot(x[:,0], x[:,1], assignments, clusters)

SSE = kmeans.score(input_fn=input_fn, steps=100)
print(SSE)

# We look for the elbow in the plot we can see it is present at 3
# Calculate the SSE scores for different values of k and then plot
K = [2, 3, 4, 5, 6]
SSe = [3365.6772, 1515.8141, 943.30, 771.155, 392.64]
plt.figure(figsize=(12,8))
plt.plot(K,SSe)
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()
