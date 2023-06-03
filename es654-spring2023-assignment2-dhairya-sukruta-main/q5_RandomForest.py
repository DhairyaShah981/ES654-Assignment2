import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import *
from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ['entropy']:
    Classifier_RF = RandomForestClassifier(6, criterion=criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    [fig1, fig2] = Classifier_RF.plot()
    fig1.savefig('plots/q5a/individual.png')
    fig2.savefig('plots/q5a/combined.png')
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))
########## RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
criteria = "squared_error"
Regressor_RF = RandomForestRegressor(6, criterion=criteria)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
[fig1, fig2] = Regressor_RF.plot()
fig1.savefig('plots/q5b/individual.png')
fig2.savefig('plots/q5b/combined.png')
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
