from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import functions as func

#import the dataset
# Pandas dataframe
data = pd.read_csv("InsuranceUserData.csv")
print(data.shape)
data.head()
headers = list(data.columns)
print(headers)
# will depend on age range the user gives
# NumPy representation of the Series object
num_prescriptions_col = data['NumPrescriptions'].values
age_range_col = data["AgeRange"].values
print("prescriptions column")
print(num_prescriptions_col)
print("Age Range")
print(age_range_col)

plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

X = np.array(list(zip(num_prescriptions_col, age_range_col)))
plt.scatter(num_prescriptions_col, age_range_col, c='black', s=7)
plt.show()
"""
# number of clusters
k = 3
# x coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)

plt.scatter(price_col, ehb_percentage_col, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.show()

# To store the values of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster labels
clusters = np.zeros(len(X))

# error function - distance between new centroids and old centroids
error = func.dist(C, C_old, None)

# Loop will run until the error becomes 0
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = func.dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = func.dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker="*", s=200, c='#050505')
plt.show() 
"""

