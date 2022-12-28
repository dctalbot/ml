from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from sklearn import svm
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [
    features_train[i][0] for i in range(0, len(features_train)) if labels_train[i] == 0
]
bumpy_fast = [
    features_train[i][1] for i in range(0, len(features_train)) if labels_train[i] == 0
]
grade_slow = [
    features_train[i][0] for i in range(0, len(features_train)) if labels_train[i] == 1
]
bumpy_slow = [
    features_train[i][1] for i in range(0, len(features_train)) if labels_train[i] == 1
]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.

clf = svm.SVC(kernel="rbf")
clf.fit(features_train, labels_train)  # define the decision surface
pred = clf.predict(features_test)

count = 0
for i, x in enumerate(pred):
    if x == labels_test[i]:
        count += 1
print(count / len(pred))

print(accuracy_score(pred, labels_test))

prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())
