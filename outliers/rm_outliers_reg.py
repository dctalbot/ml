import numpy
import matplotlib.pyplot as plt
import joblib
from sklearn import linear_model


def outlierCleaner(predictions, ages, net_worths):
    """
    Clean away the 10% of points that have the largest
    residual errors (difference between the prediction
    and the actual net worth).

    Return a list of tuples named cleaned_data where
    each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    errors = (predictions - net_worths) ** 2
    cleaned_data = list(zip(ages, net_worths, errors))
    cleaned_data.sort(key=lambda x: x[2][0])
    cleaned_data = cleaned_data[: int(len(cleaned_data) * 0.9)]

    return cleaned_data


### this data includes some outliers
ages = joblib.load(open("./practice_outliers_ages.pkl", "rb"))
net_worths = joblib.load(open("./practice_outliers_net_worths.pkl", "rb"))


### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))
from sklearn.model_selection import train_test_split

ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(
    ages, net_worths, test_size=0.1, random_state=42
)

reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)
print("slope:", reg.coef_)
print("intercept:", reg.intercept_)
print("score:", reg.score(ages_test, net_worths_test))

try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
# plt.show()


### identify and remove the most egregious outliers
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)
except NameError:
    print("Your regression object doesn't exist, or isn't name reg")
    print("Can't make predictions to use in identifying outliers")


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print("slope:", reg.coef_)
        print("intercept:", reg.intercept_)
        print("score:", reg.score(ages_test, net_worths_test))
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print("You don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    # plt.show()


else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")
