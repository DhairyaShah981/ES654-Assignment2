import pandas as pd
import math
import numpy as np


def var_weighted(Y, Weights = None, W_ = 0):
    if(W_ == 0):
        return np.var(Y)
    average = np.average(Y, weights=Weights)
    # Fast and numerically precise:
    variance = np.average((Y-average)**2, weights=Weights)
    return (variance)



def find_prob(Y: pd.Series, Weights: pd.Series):
    dict1 = {1: Y, 2: Weights}
    df = pd.concat(dict1, axis=1)
    uni_attr = Y.unique()  # collection of unique attribute values
    sum_tot = dict1[2].sum()
    probabilities = []
    for val in uni_attr:
        df1 = df[df[1] == val]
        sum1 = df1[2].sum()
        probabilities.append(sum1/sum_tot)
    return(probabilities)


def entropy(Y: pd.Series, weights: pd.Series = None, W_ = 0) -> float:
    """
    Function to calculate the entropy
    """
    # Entropy is the sum of the negative probability times the log of the probability for each class
    # Values are the distinct classes, counts are the number of instances of each class
    if(W_ == 0):
        value, counts = np.unique(Y, return_counts=True)
        # Probabilities are the fraction of instances of each class
        probabilities = counts / len(Y)
        entropy = 0  # initialising the value of entropy to zero
    else:
        probabilities = find_prob(Y, weights)    
    for probability in probabilities:
        entropy -= float(probability) * math.log(probability, 2)
    # We have defined entropy in log base 2 as taught in class
    return entropy
    pass


def gini_index(Y: pd.Series, weights: pd.Series = None, W_ = 0) -> float:
    """
    Function to calculate the gini index
    """
    # Gini index is 1 - the sum of the probability squared for each class
    # Values are the distinct classes, counts are the number of instances of each class
    if(W_== 0):
        value, counts = np.unique(Y, return_counts=True)
        probabilities = counts / len(Y)
    else: 
        probabilities = find_prob(Y, weights) 
    gini_index = 1
    gini_index -= sum(probability ** 2 for probability in probabilities)
    return gini_index
    pass


def information_gain(Y: pd.Series, attr: pd.Series, criterion: "information_gain", Weight: pd.Series = None, W_ = 0):
    """
    Function to calculate the information gain
    """
    # creating a dataframe with the column having the first output and the second column having the values of the attribute series
    we_var = 1
    if(W_== 0):
        we_var = 0
    if(we_var == 0):
        dict1 = {1: Y, 2: attr}
        df = pd.concat(dict1, axis=1)
    else:
        dict1 = {1: Y, 2: attr, 3: Weight}
        df = pd.concat(dict1, axis=1)

    si = len(Y)
    # in the case of Discrete input and discrete output, and information gain as the criterion
    if(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "information_gain"):
        uni_attr = attr.unique()  # collection of unique attribute values
        entro1 = entropy(Y, Weight, W_)  # entropy of the total dataset
        entro2 = 0
        for attri in uni_attr:
            # spilting the dataframe according to the attribute values
            df1 = df[df[2] == attri]
            # creating a series with the respective outputs of the selected attribute
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            wi_ = None
            si_ = len(y_)
            if(we_var == 1):
                wi_ = df1.iloc[:, 2].reset_index(drop=True)
                si = df[3].sum()
                si_ = df1[3].sum()
            entro2 += entropy(y_, wi_, W_)*(si_)  # adding to the entropy
        return((False, entro1 - (entro2/si)))
    # in the case of Discrete input and discrete output, and gini index as the criterion
    elif(Y.dtype.name == "category" and attr.dtype.name == "category" and criterion == "gini_index"):
        uni_attr = attr.unique()
        gini_1 = gini_index(Y, Weight, W_)  # Gini Index of the total dataset
        gini_2 = 0  # initialising the value of the gini index of the sub tables to zero
        for attri in uni_attr:
            df1 = df[df[2] == attri]
            # creating a series with the output of the corresponding attribute value
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            wi_ = None
            si_ = len(y_)
            if(we_var == 1):
                wi_ = df1.iloc[:, 2].reset_index(drop=True)
                si = df[3].sum()
                si_ = df1[3].sum()
            gini_2 += gini_index(y_, wi_, W_)*(si_)  # updating the gini_index
        return((False, gini_1 - (gini_2/si)))
    # in case of a real input and discrete output, and gini_index as the measure
    elif(Y.dtype.name == "category" and attr.dtype.name != "category" and criterion != "information_gain"):
        # sorting the dataframe according to the attribute values
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf  # initialising the max_value
        # iterating through the attribute values in order to find the split which gives the maximum gain
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            split = float(sor_dafram[2][ind] + sor_dafram[2][ind-1])/2
            # spliting the dataframe into 2 based on the split value and making the corresponding output series
            df1 = sor_dafram[sor_dafram[2] <= split].reset_index(drop=True)
            df2 = sor_dafram[sor_dafram[2] > split].reset_index(drop=True)
            y_1 = df1.iloc[:, 0].reset_index(drop=True)
            wi_1 = None
            y_2 = df2.iloc[:, 0].reset_index(drop=True)
            wi_2 = None
            si_1 = len(y_1)
            si_2 = len(y_2)
            si = len(Y)
            if(we_var == 1):
                wi_1 = df1.iloc[:, 2].reset_index(drop=True)
                wi_2 = df2.iloc[:, 2].reset_index(drop=True)
                si = df[3].sum()
                si_1 = df1[3].sum()
                si_2 = df2[3].sum()
            gini_1 = (gini_index(y_1, wi_1, W_)*(si_1) +
                      gini_index(y_2, wi_2, W_)*(si_2))/si
            val = gini_index(Y, Weight, W_) - gini_1
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    # in case of real input and discrete output, and entropy measure
    elif(Y.dtype.name == "category" and attr.dtype.name != "category" and criterion == "information_gain"):
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf  # initialising the max_value
        # iterating through the attribute values in order to find the split which gives the maximum gain
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            # spliting the dataframe into 2 based on the split value and making the corresponding output series
            split = float(sor_dafram[2][ind] + sor_dafram[2][ind-1])/2
            df1 = sor_dafram[sor_dafram[2] <= split].reset_index(drop=True)
            df2 = sor_dafram[sor_dafram[2] > split].reset_index(drop=True)
            y_1 = df1.iloc[:, 0].reset_index(drop=True)
            y_2 = df2.iloc[:, 0].reset_index(drop=True)
            wi_1 = None
            wi_2 = None
            si_1 = len(y_1)
            si_2 = len(y_2)
            si = len(Y)
            if(we_var == 1):
                wi_1 = df1.iloc[:, 2].reset_index(drop=True)
                wi_2 = df2.iloc[:, 2].reset_index(drop=True)
                si = df[3].sum()
                si_1 = df1[3].sum()
                si_2 = df2[3].sum()
            entro_1 = (entropy(y_1, wi_1, W_)*(si_1) +
                       entropy(y_2, wi_2, W_)*(si_2))/si
            val = entropy(Y, wi_, W_) - entro_1
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    # in case of Discrete input and real output, variance will be used in this case
    elif(Y.dtype.name != "category" and attr.dtype.name == "category"):
        uni_attr = attr.unique()
        var_1 = var_weighted(Y, Weight, W_)  # Gini Index of the total dataset
        var_2 = 0
        li = len(Y)
        # iterating and finding the variance of each subtable of the corresponding attribute
        for attri in uni_attr:
            df1 = df[df[2] == attri]
            y_ = df1.iloc[:, 0].reset_index(drop=True)
            wi_ = None
            li_ = len(y_)
            if(we_var == 1):
                wi_ = df1.iloc[:, 2].reset_index(drop=True)
                li_ = df1[3].sum()
                li = df[3].sum()
            var_2 += (var_weighted(y_,wi_, W_))*(li_)
        return((False, var_1 - (var_2/li)))
    # in case of real input and real output, variance will be used as a measure
    elif(Y.dtype.name != "category" and attr.dtype.name != "category"):
        sor_dafram = df.sort_values(2).reset_index(drop=True)
        best_split = 0
        max_val = -np.inf
        # iterating through the index values to find the best split
        for ind in sor_dafram.index:
            if ind == 0:
                continue
            split = float((sor_dafram[2][ind] + sor_dafram[2][ind-1])/2)
            df1 = sor_dafram[sor_dafram[2] <= split]
            df2 = sor_dafram[sor_dafram[2] > split ]
            y_1 = df1.iloc[:, 0].reset_index(drop=True)
            y_2 = df2.iloc[:, 0].reset_index(drop=True)
            wi_1 = None
            wi_2 = None
            si_1 = len(y_1)
            si_2 = len(y_2)
            si = len(Y)
            if(we_var == 1):
                wi_1 = df1.iloc[:, 2].reset_index(drop=True)
                wi_2 = df2.iloc[:, 2].reset_index(drop=True)
                si = df[3].sum()
                si_1 = df1[3].sum()
                si_2 = df2[3].sum()
            var_2 = (var_weighted(y_1,wi_1, W_)*(si_1) + var_weighted(y_2, wi_2, W_)*(si_2))/si
            val = var_weighted(Y, Weight, W_) - var_2
            if(val > max_val):
                max_val = val
                best_split = split
        return((best_split, max_val))
    pass
