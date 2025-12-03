#!/usr/bin/python

import sys
from time import time
sys.path.append("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

print(len(features_train[0]))

clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("accuracy:", accuracy)
