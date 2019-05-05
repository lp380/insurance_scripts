from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functions as func
from operator import itemgetter
from collections import Counter


plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('ggplot')

kmeans_global = None

def from_external():
    return "here from k means!!!"

def get_initial_data():
    return pd.read_csv("InsuranceUserData.csv")

def intialize_data():
    data = pd.read_csv("InsuranceUserData.csv")
    #print(data.shape)
    #print(data.head())
    headers = list(data.columns)
    #print(headers)
    # NumPy representation of the Series object
    num_prescriptions_col = data['NumPrescriptions'].values
    age_range_col = data["AgeRange"].values
    average_income_col = data["AvgIncome"].values
    print(type(average_income_col)) 
    X = np.array(list(zip(num_prescriptions_col, age_range_col, average_income_col)))
    return X


def elbow_curve_method(data):
    distorsions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distorsions.append(kmeans.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(2, 20), distorsions)
    plt.grid(True)
    plt.title('Elbow curve')
    plt.show()

def kmeans_on_insurance_data(X):
    k = 18
    # Number of clusters
    kmeans = KMeans(n_clusters=18)
    #Fitting the input data
    kmeans = kmeans.fit(X)
    # Getting the cluster labels 
    labels = kmeans.predict(X)
    # Centroid values 
    centroids = kmeans.cluster_centers_ 
    # What cluster each point belongs to.  Array of the cluster assignment for each training vector
    labels = kmeans.labels_
    counter = Counter(kmeans.labels_)
    sorted_data = sorted(counter.items(), key=itemgetter(0))
    print("size of training data", len(X))
    print("size of cluster assignments", len(kmeans.labels_))
    #print("data frame of initial data\n", data)
    for index, row  in get_initial_data().iterrows():
        print("row" + str(index) + " is in cluster " + str(kmeans.labels_[index])) 
        print(row.axes)
        #print(row["AvgIncome"])

    global kmeans_global
    kmeans_global = kmeans


# [prescriptions, AgeRange, AvgIncome]
def predict_new_point(new_point):
    # predict new point
    new_point = [5, 25, 55000]
    # get cluster it is in
    print(np.array(new_point))
    new_point_label = kmeans_global.predict(np.array((new_point)).reshape(1, -1))
    print(new_point_label)
    
if __name__ == '__main__':
    print("main method")
    data_for_algorithm = intialize_data()
    #elbow_curve_method(data_for_algorithm)
    kmeans_on_insurance_data(data_for_algorithm)
    new_point = [5, 25, 55000]
    predict_new_point(new_point)



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
