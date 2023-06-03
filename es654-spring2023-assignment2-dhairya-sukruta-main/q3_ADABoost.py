import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from metrics import *
from ensemble.ADABoost import AdaBoostClassifier
from sklearn.datasets import make_classification

# Or you could import sklearn DecisionTree

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################
# Using sklearn DecisionTree
# Q3(a)
N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
criteria = "information_gain"
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(
    base_estimator=tree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
fig1.savefig('plots/q3a/adaboost_Individual.png')
fig2.savefig('plots/q3a/adaboost_Combined.png')
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))


##################################################################################
# Q3(b)
# Generate the dataset
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=2, class_sep=0.5)
# Training on 1 decision stump of sklearn DecisionTree
X = pd.DataFrame(X)
y = pd.Series(y, dtype="category")

tree = DecisionTreeClassifier(criterion="entropy", max_depth=1)
tree.fit(X, y)
y_hat = tree.predict(X)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))

# Training on AdaBoostClassifier using 3 estimators of sklearn DecisionTree
adatree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(
    base_estimator=adatree, n_estimators=n_estimators)
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
fig1.savefig('plots/q3b/adaboost_Individual.png')
fig2.savefig('plots/q3b/adaboost_Combined.png')
print("Criteria :", "entropy")
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
