import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import DecisionTree
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# Or use sklearn decision tree
########### GradientBoostedClassifier ###################
X, y = make_regression(
    n_features=3,
    n_informative=3,
    noise=10,
    tail_strength=10,
    random_state=42,
)
#print(len(y))
d = GradientBoostedRegressor()
print(y)
X = pd.DataFrame(X)
y = pd.Series(y)

# fit
d.fit(X[:70], y[:70])

y_ = d.predict(X[70:].reset_index(drop=True))
print("the rmse of the prediction is:", rmse(
    y_, y[70:].reset_index(drop=True)))
