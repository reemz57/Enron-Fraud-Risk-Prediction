#!/usr/bin/python

import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

features_list = ["poi", "salary"]

data = featureFormat(
    data_dict,
    features_list,
    sort_keys="../tools/python2_lesson14_keys.pkl"
)

labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42
)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print(sum(pred))
print(len(pred))
print(accuracy_score(labels_test, pred))

true_positives = sum(
    int(p == 1 and t == 1) for p, t in zip(pred, labels_test)
)
print(true_positives)

print(precision_score(labels_test, pred))
print(recall_score(labels_test, pred))
