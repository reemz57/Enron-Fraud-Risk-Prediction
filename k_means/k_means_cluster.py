#!/usr/bin/python

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

data_dict = joblib.load(open("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/final_project/final_project_dataset.pkl", "rb"))

data_dict.pop("TOTAL", 0)

feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

for f1, f2, _ in finance_features:
    plt.scatter(f1, f2, _)
plt.show()

from sklearn.cluster import KMeans

n_clusters = 2
kmean = KMeans(n_clusters=n_clusters)
kmean.fit(finance_features)
pred = kmean.predict(finance_features)

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
