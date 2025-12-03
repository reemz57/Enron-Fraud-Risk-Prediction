
import os
import sys
import joblib
sys.path.append("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = joblib.load( open("final_project/final_project_dataset_modified.pkl", "rb") )


features_list = ["bonus", "salary"]
sort_keys = 'C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/tools/python2_lesson06_keys.pkl'
data = featureFormat( dictionary, features_list, remove_any_zeroes=True, sort_keys = 'tools/python2_lesson06_keys.pkl')
target, features = targetFeatureSplit( data )

from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "b"


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(feature_train ,target_train)
test_color="r"
intercept=reg.intercept_
slope=reg.coef_

print(intercept)
print(slope)

print(reg.score(feature_test,target_test))



import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
print(reg.coef_)
plt.plot(feature_train, reg.predict(feature_train), color="r")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
