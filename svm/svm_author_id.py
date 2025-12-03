#!/usr/bin/python3

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

clf = SVC(kernel="rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("prediction time:", round(time() - t0, 3), "s")

accuracy = accuracy_score(labels_test, pred)
print("accuracy:", accuracy)

print(pred[10])
print(pred[26])
print(pred[50])
