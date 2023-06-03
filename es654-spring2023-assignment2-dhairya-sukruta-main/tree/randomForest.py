import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as sktree
import sys
from tree.base import DecisionTree
from base import DecisionTree


class RandomForestClassifier():
    def __init__(self, n_estimators=100, base_estimator=DecisionTree, criterion='gini', randomFeat=2):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.estimators = []
        self.X = None
        self.y = None
        self.randomFeat = randomFeat
        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        m = self.randomFeat
        for i in range(self.n_estimators):
            bootstrap_sample = X.sample(n=len(X), replace=True)
            bootstrap_label = y.loc[bootstrap_sample.index].reset_index(
                drop=True)
            bootstrap_sample = bootstrap_sample.reset_index(drop=True)

            subset_x = bootstrap_sample.sample(m, axis='columns')
            randomTree = self.base_estimator(
                criterion=self.criterion, forest=1, m=m)
            randomTree.fit(bootstrap_sample, bootstrap_label)
            self.estimators.append(randomTree)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_preds = np.array([estimator.predict(X)
                           for i, estimator in enumerate(self.estimators)])
        y_pred = pd.Series([np.bincount(y_preds[:, i]).argmax()
                           for i in range(y_preds.shape[1])], index=X.index)
        return y_pred
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        # 1

        for i, estimator in enumerate(self.estimators):
            # printing the estimators and writing the output to a file
            with open(f'estimator_{i}.txt', 'w') as f:
                # redirect the output to the file
                sys.stdout = f
            # call the plot method to print the output to the file
                estimator.plot()
            # reset the standard output to the console
            sys.stdout = sys.__stdout__

        # 2

        fig1, axes1 = plt.subplots(1, self.n_estimators, figsize=(20, 20))
        # list of all features in the dataset
        featureSet = list(self.X.columns)
        # select two random features from featureSet

        for i, estimator in enumerate(self.estimators):
            ind1, ind2 = np.random.choice(
                range(len(featureSet)), 2, replace=False)
            feat = [featureSet[ind1], featureSet[ind2]]
            features = self.X.columns
            x_min, x_max = self.X.iloc[:, feat[0]].min(
            ) - 0.5, self.X.iloc[:, feat[0]].max() + 0.5
            y_min, y_max = self.X.iloc[:, feat[1]].min(
            ) - 0.5, self.X.iloc[:, feat[1]].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
            cols1 = set(mesh.columns)
            cols2 = set(self.X.columns)
            for col in cols2 - cols1:
                mesh[col] = self.X[col].mean()
            y_hat = estimator.predict(mesh).values.reshape(xx.shape)

            plot = axes1[i].contourf(
                xx, yy, y_hat, cmap=plt.cm.RdYlBu, alpha=0.5)
            fig1.colorbar(plot, ax=axes1[i], shrink=0.8)
            for j, class_value in enumerate(self.y.unique()):
                axes1[i].scatter(self.X[self.y == class_value][feat[0]], self.X[self.y == class_value]
                                 [feat[1]], label=class_value, s=60, c='rgwyb'[j], edgecolors='black')

            axes1[i].set_title(f"Decision surface of tree {i}")
            axes1[i].legend()
            axes1[i].set_xlabel(feat[0])
            axes1[i].set_ylabel(feat[1])

# 3
        fig2, axes2 = plt.subplots(1, 1, figsize=(20, 20))

        # list of all features in the dataset
        featureSet = list(self.X.columns)
        # select two random features from featureSet
        ind1, ind2 = np.random.choice(range(len(featureSet)), 2, replace=False)
        featCombined = [featureSet[ind1], featureSet[ind2]]
        x_min, x_max = self.X.iloc[:, featCombined[0]].min(
        ) - 0.5, self.X.iloc[:, featCombined[0]].max() + 0.5
        y_min, y_max = self.X.iloc[:, featCombined[1]].min(
        ) - 0.5, self.X.iloc[:, featCombined[1]].max() + 0.5
        xaxis, yaxis = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                   np.arange(y_min, y_max, 0.1))
        mesh = pd.DataFrame(np.c_[xaxis.ravel(), yaxis.ravel()])
        cols1 = set(mesh.columns)
        cols2 = set(self.X.columns)
        for col in cols2 - cols1:
            mesh[col] = self.X[col].mean()
        y_final = self.predict(mesh).values.reshape(xaxis.shape)
        plot1 = axes2.contourf(xaxis, yaxis, y_final,
                               cmap=plt.cm.RdYlBu, alpha=0.5)
        fig2.colorbar(plot1, ax=axes2, shrink=0.8)
        for j, class_value in enumerate(self.y.unique()):
            axes2.scatter(self.X[self.y == class_value][featCombined[0]], self.X[self.y == class_value]
                          [featCombined[1]], label=class_value, s=60, c='rgwyb'[j], edgecolors='black')
        axes2.set_xlabel("Attribute1")
        axes2.set_ylabel("Attribute2")
        axes2.legend()
        axes2.set_title("Combined Decision Surface")
        return fig1, fig2


class RandomForestRegressor():
    def __init__(self, n_estimators=100, base_estimator=DecisionTree, criterion='squared_error', max_attributes=4, randomFeat=2):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_attributes = max_attributes
        self.estimators = []
        self.X = None
        self.y = None
        self.randomFeat = randomFeat
        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        m = self.randomFeat
        for i in range(self.n_estimators):
            bootstrap_sample = X.sample(n=len(X), replace=True)
            bootstrap_label = y[bootstrap_sample.index].reset_index(drop=True)
            bootstrap_sample = bootstrap_sample.reset_index(drop=True)
            # make subset x with features between 1 and max_attributes
            subset_x = bootstrap_sample.sample(m, axis='columns')
            randomTree = self.base_estimator(
                criterion=self.criterion, forest=1, m=m)
            randomTree.fit(bootstrap_sample, bootstrap_label)
            self.estimators.append(randomTree)
        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_preds = np.array([estimator.predict(X)
                           for i, estimator in enumerate(self.estimators)])
        y_pred = pd.Series([np.mean(y_preds[:, i])
                           for i in range(y_preds.shape[1])])
        return y_pred
        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        # 1

        for i, estimator in enumerate(self.estimators):
            # printing the estimators and writing the output to a file
            with open(f'estimator_{i}.txt', 'w') as f:
                # redirect the output to the file
                sys.stdout = f
            # call the plot method to print the output to the file
                estimator.plot()
            # reset the standard output to the console
            sys.stdout = sys.__stdout__

        # 2
        # create a figure with 1 row and n_estimators columns
        # the subplots in figure should have projection='3d'
        fig1, ax = plt.subplots(1, self.n_estimators, figsize=(
            20, 20), subplot_kw={'projection': '3d'})
        # list of all features in the dataset
        featureSet = list(self.X.columns)
        # select two random features from featureSet

        for i, estimator in enumerate(self.estimators):
            ind1, ind2 = np.random.choice(
                range(len(featureSet)), 2, replace=False)
            x_min, x_max = self.X.iloc[:, featureSet[ind1]].min(
            ) - 0.5, self.X.iloc[:, featureSet[ind1]].max() + 0.5
            y_min, y_max = self.X.iloc[:, featureSet[ind2]].min(
            ) - 0.5, self.X.iloc[:, featureSet[ind2]].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
            cols1 = set(mesh.columns)
            cols2 = set(self.X.columns)

            for col in cols2 - cols1:
                mesh[col] = self.X[col].mean()
            y_hat = estimator.predict(mesh).values.reshape(xx.shape)
            # Plot the surface
            ax[i].plot_surface(xx, yy, y_hat.reshape(xx.shape),
                               cmap=plt.cm.RdYlBu, alpha=0.5)
            ax[i].scatter3D(self.X.iloc[:, featureSet[ind1]],
                            self.X.iloc[:, featureSet[ind2]], self.y, color="black")
            # ax.contour(xx, yy, Z.reshape(xx.shape), 10, offset=-1, lw=3, colors="k", linestyles="solid", alpha=0.5)
            # Set the x, y, z limits and labels
            ax[i].set_xlabel('X')
            ax[i].set_ylabel('Y')
            ax[i].set_zlabel('Z')
            ax[i].set_xlim(x_min, x_max)
            ax[i].set_ylim(y_min, y_max)
        # 3
        fig2, axes = plt.subplots(1, 1, figsize=(
            20, 20), subplot_kw={'projection': '3d'})
        ind1, ind2 = np.random.choice(range(len(featureSet)), 2, replace=False)
        x_min, x_max = self.X.iloc[:, featureSet[ind1]].min(
        ) - 0.5, self.X.iloc[:, featureSet[ind1]].max() + 0.5
        y_min, y_max = self.X.iloc[:, featureSet[ind2]].min(
        ) - 0.5, self.X.iloc[:, featureSet[ind2]].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        mesh = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
        cols1 = set(mesh.columns)
        cols2 = set(self.X.columns)

        for col in cols2 - cols1:
            mesh[col] = self.X[col].mean()
        y_final = self.predict(mesh).values.reshape(xx.shape)
        # Plot the surface
        axes.plot_surface(xx, yy, y_hat.reshape(xx.shape),
                          cmap=plt.cm.RdYlBu, alpha=0.5)
        axes.scatter3D(self.X.iloc[:, featureSet[ind1]],
                       self.X.iloc[:, featureSet[ind2]], self.y, color="black")
        # ax.contour(xx, yy, Z.reshape(xx.shape), 10, offset=-1, lw=3, colors="k", linestyles="solid", alpha=0.5)
        # Set the x, y, z limits and labels
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)

        return fig1, fig2
    pass
