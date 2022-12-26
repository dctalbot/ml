""" 
    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

# sys.path.append(".../enron/")
from enron.email_preprocess import preprocess
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

features_train, features_test, labels_train, labels_test = preprocess(
    "./enron/word_data.pkl", "./enron/email_authors.pkl"
)

clf = AdaBoostClassifier()
t0 = time()
clf = clf.fit(features_train, labels_train)
print("Training Time:", round(time() - t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time() - t0, 3), "s")

count = 0
for i, x in enumerate(pred):
    if x == labels_test[i]:
        count += 1

print(count / len(pred))
