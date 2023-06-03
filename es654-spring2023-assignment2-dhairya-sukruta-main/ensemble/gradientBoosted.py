from tree.base import *
from metrics import *

class GradientBoostedRegressor:
    def __init__(
        self, base_estimator='DTC', n_estimators=3, learning_rate=0.1
    ):  # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate
        self.li = []

    def fit(self, X, y):
        arr = [np.average(y)]*(len(y))
        y_ = pd.Series(arr)
        c = DecisionTree(max_depth=0)
        c.fit(X, y)
        self.li.append(c)
        for x in range(self.n_estimator):
            b = DecisionTree(max_depth=10)
            y_1 = y.subtract(y_)
            b.fit(X, y_1)
            y_hat = b.predict(X)
            self.li.append(b)
            y_ = y_ + self.learning_rate * y_hat

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y = self.li[0].predict(X)
        count = 0
        for x in self.li:
            if(count == 0):
                next
            count = 1
            y = y.add(self.learning_rate * x.predict(X))
        return(y)
