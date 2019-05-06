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
initial_data = None

def from_external():
    return "here from k means!!!"

def get_initial_data():
    return pd.read_csv("InsuranceUserData.csv")

def intialize_data():
    data = pd.read_csv("InsuranceUserData.csv")
    global initial_data
    initial_data = data
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
    print(initial_data[kmeans.labels_ == 10])
    
    """
    km_labels = kmeans.labels_
    index = [x[0] for x, value in np.ndenumerate(km_labels) if value ==15]
    print("index")
    print(index)
    """

    """
    for index, row  in get_initial_data().iterrows():
        print("row" + str(index) + " is in cluster " + str(kmeans.labels_[index])) 
        cluster_points = pd.DataFrame()
        cluster_points['cluster'] = labels
        cluster_points[cluster_points.cluster == 5]
        print(cluster_points)
        print(row.axes)
        #print(row["AvgIncome"])
    """
    global kmeans_global
    kmeans_global = kmeans


# [prescriptions, AgeRange, AvgIncome]
def predict_new_point(new_point):
    # predict new point
    new_point = [5, 25, 55000]
    # get cluster it is in
    print(np.array(new_point))
    new_point_label = kmeans_global.predict(np.array((new_point)).reshape(1, -1))
    print("The point belongs to cluster", new_point_label)
    dict = create_dictionary()
    range = dict[new_point_label]

def create_dictionary():

    x = {}

    x[0] = (147.84, 205.30)
    x[1] = (205.31, 262.76)
    x[2] = (262.77, 320.22)
    x[3] = (320.23, 377.68)
    x[4] = (377.69, 435.14)
    x[5] = (435.15, 492.60)
    x[6] = (492.61, 550.06)
    x[7] = (550.07, 607.52)
    x[8] = (607.53, 664.98)
    x[9] = (664.99, 722.04)
    x[10] = (722.05, 779.90)
    x[11] = (779.91, 837.06)
    x[12] = (837.07, 894.82)
    x[13] = (894.83, 952.28)
    x[14] = (952.29, 1009.74)
    x[15] = (1009.75, 1057.20)
    x[16] = (1057.21, 1124.66)
    x[17] = (1124.67, 1182.10)

    return x

    
if __name__ == '__main__':
    print("main method")
    data_for_algorithm = intialize_data()
    #elbow_curve_method(data_for_algorithm)
    kmeans_on_insurance_data(data_for_algorithm)
    new_point = [5, 25, 55000]
    predict_new_point(new_point)

