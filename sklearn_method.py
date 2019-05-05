from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

#import the dataset
data = pd.read_csv('xclara.csv')
print(data.shape)
data.head()

#Getting the values
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

# Number of clusters
kmeans = KMeans(n_clusters=3)
#Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels 
labels = kmeans.predict(X)
# Centroid values 
centroids = kmeans.cluster_centers_
print(centroids)

