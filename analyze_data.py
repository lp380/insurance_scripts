from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import functions as func

#import the dataset
# Pandas dataframe
data = pd.read_csv("insurance_data.csv")
print(data.shape)
data.head()
headers = list(data.columns)
print(headers)
# will depend on age range the user gives
# NumPy representation of the Series object
price_col = data['PremiumAdultIndividualAge27'].values
ehb_percentage_col = data["EHBPercentofTotal"].values

for i in price_col:
    if i is None:
        print("NONE")
    #print(i)
