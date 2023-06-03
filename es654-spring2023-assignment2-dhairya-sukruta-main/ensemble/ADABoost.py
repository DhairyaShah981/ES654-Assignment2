from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import weighted_mode
import numpy as np
import pandas as pd
from metrics import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdaBoostClassifier():
    # Optional Arguments: Type of estimator
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=3):
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = 1
        self.criterion = "entropy"
        self.estimators = []
        self.alphas = []
        self.classes_ = None
        self.X = None
        self.y = None
        self.weights = []
    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        
        data_weights = np.ones(len(y))/len(y)
        
        self.classes_ = np.unique(y)
        for i in range(self.n_estimators):
            self.weights.append(data_weights.copy())
            estimator = self.base_estimator(
                criterion="entropy", max_depth=self.max_depth)
            estimator.fit(X=X, y=y, sample_weight=data_weights)
            y_hat = pd.Series(estimator.predict(X))

            # finding error and alpha
            error = np.sum(data_weights[y != y_hat])/np.sum(data_weights)
            alpha = 0.5*np.log((1-error)/error)

            # updating weights
            data_weights[y == y_hat] *= np.exp(-alpha)
            data_weights[y != y_hat] *= np.exp(alpha)
            data_weights /= np.sum(data_weights)

            # storing alpha and estimator
            self.alphas.append(alpha)
            self.estimators.append(estimator)

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        """
        n_classes = len(self.classes_)

        # Initialize predictions to 0
        preds = np.zeros((X.shape[0], n_classes))

        # Loop through the estimators and update the predictions
        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.estimators)):
            proba = tree.predict_proba(X)
            preds += proba * alpha_m  # update preds

        # Find the class with maximum prediction for each sample
        y_hat = np.argmax(preds, axis=1)

        return pd.Series(y_hat)
    def predict_proba(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y_proba: pd.DataFrame with rows corresponding to output variable. The output variable in a row is the probability prediction for sample in corresponding row in X.
        """
        n_classes = len(self.classes_)

        # Initialize predictions to 0
        preds = np.zeros((X.shape[0], n_classes))

        # Loop through the estimators and update the predictions
        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.estimators)):
            proba = tree.predict_proba(X)
            preds += proba * alpha_m  # update preds

        # Normalize the predictions to get probabilities
        preds /= preds.sum(axis=1).reshape(-1, 1)

        return preds

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        
        # creating the grid
        h = 0.1
        x_min, x_max = self.X.min()[0] - 1, self.X.max()[0] + 1
        y_min, y_max = self.X.min()[1] - 1, self.X.max()[1] + 1
        Attr_X1, Attr_X2 = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # create the figure1
        fig1, ax1 = plt.subplots(1, self.n_estimators, figsize=(15, 5))

        for i, (alpha, estimator) in enumerate(zip(self.alphas, self.estimators)):
            # predict on grid
            y_hat = estimator.predict(np.c_[Attr_X1.ravel(), Attr_X2.ravel()]).reshape(Attr_X1.shape)

            # scatter plot of the data with weight as size
            ax1[i].scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], s=500*self.weights[i], c=self.y, cmap=ListedColormap(['red', 'blue']))

            # plot the decision boundary
            ax1[i].contourf(Attr_X1, Attr_X2, y_hat, alpha=0.5, cmap=ListedColormap(['red', 'blue']))
            ax1[i].set_title(f'Estimator {i+1} - alpha = {alpha:.3f}')

        # create figure2
        fig2, ax2 = plt.subplots(figsize=(8, 6))

        # get predictions on the grid
        y_preds = self.predict(np.c_[Attr_X1.ravel(), Attr_X2.ravel()]).values.reshape(Attr_X1.shape)


        # plot decision boundary
        ax2.contourf(Attr_X1, Attr_X2, y_preds, alpha=0.5, cmap=ListedColormap(['red', 'blue']))

        # scatter plot the data with size as weight
        ax2.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], s=30, c=self.y, cmap=ListedColormap(['red', 'blue']))

        # set title and show the plot
        ax2.set_title('Combined Estimators')
        plt.show()

        return [fig1, fig2]
        pass
