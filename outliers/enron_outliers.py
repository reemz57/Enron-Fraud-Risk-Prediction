
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


salary = data[:, 0]  # First column is salary
bonus = data[:, 1]   # Second column is bonus

# the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(salary, bonus, alpha=0.5) 
plt.title("Scatter Plot of Salary vs Bonus")
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.grid(True)
plt.show() 

