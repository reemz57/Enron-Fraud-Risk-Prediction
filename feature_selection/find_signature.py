import joblib
import numpy
numpy.random.seed(42)

words_file = "C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/text_learning/your_word_data.pkl"
authors_file = "C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master/text_learning/your_email_authors.pkl"

word_data = joblib.load(open(words_file, "rb"))
authors = joblib.load(open(authors_file, "rb"))

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    word_data, authors, test_size=0.1, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, stop_words="english"
)
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)
print(accuracy_score(labels_test, pred))

importances = clf.feature_importances_

for i, imp in enumerate(importances):
    if imp > 0.2:
        print(i, imp, vectorizer.get_feature_names_out()[i])
