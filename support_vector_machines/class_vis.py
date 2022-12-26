# from udacityplots import *
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

# import numpy as np
# import matplotlib.pyplot as plt
# plt.ioff()


def prettyPicture(clf, X_test, y_test):
    x_start = 0.0
    x_stop = 1.0
    y_start = 0.0
    y_stop = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_start, m_stop]x[y_start, y_stop].
    step = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_start, x_stop, step), np.arange(y_start, y_stop, step)
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot the test points
    grade_sig = [X_test[i][0] for i in range(0, len(X_test)) if y_test[i] == 0]
    bumpy_sig = [X_test[i][1] for i in range(0, len(X_test)) if y_test[i] == 0]
    grade_bkg = [X_test[i][0] for i in range(0, len(X_test)) if y_test[i] == 1]
    bumpy_bkg = [X_test[i][1] for i in range(0, len(X_test)) if y_test[i] == 1]

    plt.scatter(grade_sig, bumpy_sig, color="b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")


import base64
import json
import subprocess


def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data["name"] = name
    data["format"] = format
    data["bytes"] = str(base64.encodebytes(bytes))
    print(image_start + json.dumps(data) + image_end)
