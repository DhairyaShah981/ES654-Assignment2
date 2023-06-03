from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class BaggingClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=100, n_jobs=-1):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.criterion = "entropy"
        self.estimators = []
        self.datasets = []
        self.X = None
        self.y = None
        self.n_jobs = n_jobs
        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y

        for i in range(self.n_estimators):
            bootstrap_sample = X.sample(n=len(X), replace=True)
            bootstrap_label = y[bootstrap_sample.index].reset_index(drop=True)
            bootstrap_sample = bootstrap_sample.reset_index(drop=True)
            self.datasets.append([bootstrap_sample, bootstrap_label])
        self.estimators = Parallel(n_jobs=self.n_jobs)(delayed(self.base_estimator().fit)(
            X=dataset[0], y=dataset[1]) for dataset in self.datasets)

        pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        for tree in self.estimators:
            predictions.append(tree.predict(X))
        final_preds = pd.DataFrame(predictions).mode(axis=0).iloc[0]
        return final_preds
        pass

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
    
        featureSet = list(self.X.columns)
        # select random two features from featureSet
        feat = np.random.choice(featureSet, 2, replace=False)
        h = 0.1
        x_min, x_max = self.X.min()[feat[0]] - 1, self.X.max()[feat[0]] + 1
        y_min, y_max = self.X.min()[feat[1]] - 1, self.X.max()[feat[1]] + 1
        Attr_X1, Attr_X2 = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Create a dataframe for storing the predicted outputs for each estimator
        y_final = pd.DataFrame()

        # Create figure 1 with `n_estimators` subplots
        fig1, ax1 = plt.subplots(1, self.n_estimators, figsize=(20, 5))

        # Plot the decision surface for each estimator and store its predicted outputs
        for i, tree in enumerate(self.estimators):
            y_hat = tree.predict(pd.concat(
                [pd.Series(Attr_X1.flatten()), pd.Series(Attr_X2.flatten())], axis=1))
            y_hat = y_hat.reshape(Attr_X1.shape)
            y_final[i] = pd.Series(y_hat.reshape(len(y_hat.flatten())))

            # Plot the decision surface for this estimator
            fig1.colorbar(ax1[i].contourf(Attr_X1, Attr_X2,
                                          y_hat, cmap=plt.cm.YlOrRd), ax=ax1[i], shrink=0.5)
            ax1[i].scatter(self.X.iloc[:, 0], self.X.iloc[:, 1],
                           c=self.y, cmap=plt.cm.YlOrRd)
            ax1[i].set_title("Round No: " + str(i+1))
            ax1[i].set_xlabel("Attribute 1")
            ax1[i].set_ylabel("Attribute 2")

        # Create figure 2 with a single plot
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))

        # Find the predicted outputs with the highest frequency across all estimators
        y_final = y_final.mode(axis=1)[0]

        # Plot the decision surface for the combined output of all estimators
        y_final = y_final.values.reshape(Attr_X1.shape)
        fig2.colorbar(ax2.contourf(Attr_X1, Attr_X2, y_final,
                      cmap=plt.cm.YlOrRd), ax=ax2, shrink=0.5)
        ax2.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1],
                    c=self.y, cmap=plt.cm.YlOrRd)
        ax2.set_title("Combined")
        ax2.set_xlabel("Attribute 1")
        ax2.set_ylabel("Attribute 2")
        return [fig1, fig2]
        pass
