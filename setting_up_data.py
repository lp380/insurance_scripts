import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

input_file = "olympics2016.csv"

df = pd.read_csv(input_file, header=0)
# df.describe()
print(df)

original_headers = list(df.columns)

# print('[%s]' % ', '.join(map(str, original_headers)))

print(list(pd.DataFrame.to_numpy(df.columns)))

# put into scikit learn
# get just the numbers
dfEdited = df.drop(['NOC', 'Country Code'], axis=1)
print("dropped off first 2 columns")
print(dfEdited)
kmeans = KMeans(n_clusters=2)
kmeans.fit(dfEdited)
#print(kmeans)
dfEdited.describe()
# see how it did
# plot k means result




