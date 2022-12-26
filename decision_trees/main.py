from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

count = 0
for i, x in enumerate(pred):
    if x == labels_test[i]:
        count += 1
print(count / len(pred))

print(accuracy_score(pred, labels_test))

prettyPicture(clf, features_test, labels_test)
