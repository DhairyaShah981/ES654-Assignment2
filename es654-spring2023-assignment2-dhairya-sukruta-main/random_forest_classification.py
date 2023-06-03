import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

# Write code here

from sklearn.datasets import make_classification
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
X = pd.DataFrame(X)
y = pd.Series(y, dtype="category")

# For plotting
#plt.scatter(X[:, 0], X[:, 1], c=y)
########### RandomForestClassifier ###################


for criteria in ['entropy']:
    Classifier_RF = RandomForestClassifier(4, criterion=criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    [fig1, fig2] = Classifier_RF.plot()
    fig1.savefig('plots/q5classification/individual.png')
    fig2.savefig('plots/q5classification/combined.png')
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
