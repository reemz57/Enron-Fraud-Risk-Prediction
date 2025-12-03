import os
import joblib
import re
import sys

# Replace any ".." with your required path  
BASE_PATH = r"C:/Users/reema/OneDrive - Indian Institute of Technology Guwahati/Documents/Udacity/ud120-projects-master/ud120-projects-master"

sys.path.append(os.path.join(BASE_PATH, "tools"))
from parse_out_email_text import parseOutText


"""
Processes the emails from Sara and Chris to extract the features,
clean them, remove target names, and prepare them for classification.
"""

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        temp_counter += 1

        # LIMIT to first 200 emails for faster development
        if temp_counter >= 200:
            break

        # Build full path using your required base directory
        path = os.path.join(BASE_PATH, path.strip())

        print(f"Processing: {path}")

        with open(path, "r") as email:
            # Extract raw text from email
            text = parseOutText(email)

            # Remove target words
            for word in ["sara", "shackleton", "chris", "germani"]:
                text = text.replace(word, "")

            # Store processed text
            word_data.append(text)

            # Labels: 0 = Sara, 1 = Chris
            from_data.append(0 if name == "sara" else 1)

print("Emails Processed.")

from_sara.close()
from_chris.close()

# Save processed data
joblib.dump(word_data, open("your_word_data.pkl", "wb"))
joblib.dump(from_data, open("your_email_authors.pkl", "wb"))

print("Data saved successfully.")

"""
PART 4: TF-IDF VECTORISATION
"""
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(word_data)

print("TF-IDF vectorization complete.")
print("Number of features (vocabulary size):", len(vectorizer.get_feature_names_out()))
