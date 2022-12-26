import os
import sys
import joblib

# sys.path.append(os.path.abspath("../tools/"))
from enron.feature_format import featureFormat, targetFeatureSplit
from sklearn.linear_model import LinearRegression

dictionary = joblib.load(open("./enron/final_project_dataset_modified.pkl", "rb"))


### list the features you want to look at--first item in the
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(
    dictionary,
    features_list,
    remove_any_zeroes=True,
    sort_keys="./enron/python2_lesson06_keys.pkl",
)
target, features = targetFeatureSplit(data)

### training-testing split needed in regression, just like classification
from sklearn.model_selection import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(
    features, target, test_size=0.5, random_state=42
)
train_color = "b"
test_color = "r"

reg = LinearRegression()
reg = reg.fit(feature_train, target_train)
print(reg.coef_)  # slope
print(reg.intercept_)  # y-intercept

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")


### draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass
reg.fit(feature_test, target_test)
print(reg.coef_)  # slope
plt.plot(feature_train, reg.predict(feature_train), color="b")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

# pred = reg.predict(feature_train)
score = reg.score(feature_test, target_test)
print(score)
