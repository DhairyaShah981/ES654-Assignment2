from metrics import *
import pandas as pd
import random
from tree.base import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# compare both the trees
# Generate the dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2, class_sep=0.5)

# Shuffle the dataset and split into training and test sets


X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X = pd.DataFrame(X)
y = pd.Series(y)
weight = np.random.rand(100)
weight = pd.Series(weight)


# training the decision tree over the first 70% of the dataset
f = DecisionTree()
f.fit(X[:70], y[:70], weight[:70], 1)
y_hat = f.predict(X[70:])
print("the accuracy of the prediction is (without stuffling):",
      accuracy(y_hat, y[70:].reset_index(drop=True)))


# inorder to suffle the data
Y = X
Y["results"] = y
Y = Y.sample(frac=1).reset_index(drop=True)

# creating a corresponding input dataframe and result series
y_ = pd.Series(Y["results"])
Y = Y.drop("results", axis=1)


# accuracy of the prediction with the weighted decision tree (for the suffled dataset)
g = DecisionTree()
g.fit(Y[:70], y_[:70], weight[:70], 1)
y_hat = g.predict(Y[70:])
print("the accuracy of the prediction is (with stuffling):",
      accuracy(y_hat, y_[70:].reset_index(drop=True)))


# Create a meshgrid of points to cover the entire feature space
x_min, x_max = Y.iloc[:, 0].min() - 0.5, Y.iloc[:, 0].max() + 0.5
y_min, y_max = Y.iloc[:, 1].min() - 0.5, Y.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])

# Use your trained decision tree to predict the class for each point in the meshgrid
Z = g.predict(mesh)

# Reshape the predicted classes to the same shape as the meshgrid
Z = Z.values.reshape(xx.shape)

# Plot the decision boundary and the training data
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, alpha=0.8)
plt.title("Decision Tree Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Comparing the accuracy of the tests with sklearns accuracy
clf = DecisionTreeClassifier(random_state=0)
clf.fit(Y[:70], y_[:70], sample_weight=weight[:70])
y_pred = clf.predict(Y[70:])
print("the accuracy of the prediction in sklearn is (with stuffling):",
      accuracy(y_pred, y_[70:].reset_index(drop=True)))
