from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as func
from operator import itemgetter
from collections import Counter

def from_external():
    return "here from k means!!!"


#import the dataset
# Pandas dataframe
data = pd.read_csv("InsuranceUserData.csv")
#print(data.shape)
data.head()
headers = list(data.columns)
#print(headers)
# will depend on age range the user gives
# NumPy representation of the Series object
num_prescriptions_col = data['NumPrescriptions'].values
age_range_col = data["AgeRange"].values
average_income_col = data["AvgIncome"].values
#print("prescriptions column")
#print(num_prescriptions_col)
#print("Age Range")
#print(age_range_col)

plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')
X = np.array(list(zip(num_prescriptions_col, age_range_col, average_income_col)))
print("Giving to kmeans...")
print(X)
distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
#plt.show()


k = 18
# Number of clusters
kmeans = KMeans(n_clusters=18)
#Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels 
labels = kmeans.predict(X)
# Centroid values 
centroids = kmeans.cluster_centers_
#print(centroids)
#print(labels)
counter = Counter(kmeans.labels_)
#print(counter)
sorted = sorted(counter.items(), key=itemgetter(0))
#print(sorted)

kmeans.cluster_centers_
#what cluster each point belongs to 
kmeans.labels_

# get average income and average persecitpions of each cluster
#print(kmeans.labels_)
print("labels")
# array of the cluster assignment for each training vector
print(kmeans.labels_)
print("whooaaaa")

print("size of training data", len(X))
print("size of cluster assignments", len(kmeans.labels_))

#print("data frame of initial data\n", data)

for index, row  in data.iterrows():
    print("row" + str(index) + " is in cluster " + str(kmeans.labels_[index])) 
    #print(row.axes)
    #print(row["AvgIncome"])

# predict new point
#prescriptions, AgeRange, AvgIncome
new_point = [5, 25, 55000]
# get cluster it is in
new_point_label = kmeans.predict(new_point)
print(new_point_label)

"""
dict = {}

cluster_map = pd.DataFrame()
cluster_map['data_index'] = data.index.array
cluster_map['cluster'] = kmeans.labels_
print(cluster_map[cluster_map.cluster == 3])

for i in range(0, k):
    cluster = cluster_map[cluster_map.cluster == i]
    print(cluster)
"""    
