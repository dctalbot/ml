""" 
    Use a SVM to identify emails from the Enron corpus by their authors:
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn import svm

# sys.path.append(".../enron/")
from enron.email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess(
    "./enron/word_data.pkl", "./enron/email_authors.pkl"
)


features_train = features_train[: len(features_train) // 100]
labels_train = labels_train[: len(labels_train) // 100]

clf = svm.SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time() - t0, 3), "s")

t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time() - t0, 3), "s")

count = 0
for i, x in enumerate(pred):
    if x == labels_test[i]:
        count += 1

print(count / len(pred))

print(len(pred))

print(len([x for x in pred if x == 1]))

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
