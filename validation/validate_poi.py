#!/usr/bin/python

import os
import joblib
import sys

BASE_PATH = r"C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master"
sys.path.append(os.path.join(BASE_PATH, "tools"))
from feature_format import featureFormat, targetFeatureSplit

data_path = os.path.join(BASE_PATH, "final_project/final_project_dataset.pkl")
data_dict = joblib.load(open(data_path, "rb"))

features_list = ["poi", "salary"]
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.tree import DecisionTreeClassifier
clf_overfit = DecisionTreeClassifier()
clf_overfit.fit(features, labels)
print(clf_overfit.score(features, labels))

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

clf_validated = DecisionTreeClassifier()
clf_validated.fit(features_train, labels_train)
print(clf_validated.score(features_test, labels_test))
