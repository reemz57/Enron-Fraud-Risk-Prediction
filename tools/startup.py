print("Checking for nltk")
try:
    import nltk
except ImportError:
    print("You should install nltk before continuing")

print("Checking for numpy")
try:
    import numpy
except ImportError:
    print("You should install numpy before continuing")

print("Checking for scipy")
try:
    import scipy
except:
    print("You should install scipy before continuing")

print("Checking for sklearn")
try:
    import sklearn
except:
    print("You should install sklearn before continuing")

# Remove the downloading section
# print("Downloading the Enron dataset (this may take a while)")
# print("To check on progress, you can cd up one level, then execute <ls -lthr>")
# print("Enron dataset should be last item on the list, along with its current size")
# print("Download will complete at about 1.82 GB")

# Set the path to your local extracted data
enron_data_path = "C:\\Users\\reema\\OneDrive - Indian Institute of Technology Guwahati\\Documents\\Udacity\\maildir"

# You can add any additional code here to work with the data in `enron_data_path`

print("You're ready to go!")
