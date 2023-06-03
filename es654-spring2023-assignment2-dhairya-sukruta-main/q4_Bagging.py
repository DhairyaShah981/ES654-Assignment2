import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import time
from metrics import *
from metrics import *

from ensemble.bagging import BaggingClassifier


# Or use sklearn decision tree

########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")

criteria = "information_gain"
tree = DecisionTreeClassifier
begin = time.time()
Classifier_B = BaggingClassifier(
    base_estimator=tree, n_estimators=n_estimators, n_jobs = 1)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
end = time.time()
[fig1, fig2] = Classifier_B.plot()
fig1.savefig('plots/q4/Bagging_Individual.png')
fig2.savefig('plots/q4/Bagging_Combined.png')
print("Time taken:", end - begin)
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))

import matplotlib.pyplot as plt

n_jobs = [-1, 1, 2, 3, 4, 5, 6, 7, 8]
time = [4.757517337799072, 0.018538713455200195, 2.992455244064331, 3.1556577682495117, 3.810579538345337, 3.4977099895477295, 1.4183261394500732, 0.02974081039428711, 1.1476285457611084]

plt.plot(n_jobs, time, 'bo-')
plt.xlabel('n_jobs')
plt.ylabel('time')
plt.title('n_jobs vs time')
plt.show()
