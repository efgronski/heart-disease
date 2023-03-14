"""
Machine Learning Analysis

This file contains the functions used to analyze our heart disease
data using the KNeighborsClassifier model from the sklearn library.
Coding analysis for these functions can be found within the Jupyter
notebooks that have the same coding process within them, with output
next to processes for an easier reading.
"""

# Import libraries
import pandas as pd
import numpy as np
import statsmodels as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


def load_in_data(filename: str) -> pd.DataFrame:
    """
    Loads in heart disease data as a pandas dataframe
    """
    return pd.read_csv(filename)


def remove_na(data: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in the data and removes the NA values
    """
    new_data = data.dropna()
    return new_data


def fit_kneighbors_to_num(data: pd.DataFrame) -> confusion_matrix:
    """
    Takes heart data and creates a KNeighborsClassifier model. The
    predicted value is num which comes from the heart data. This function
    then runs it through a range of 1 to 25 neighbors to find the
    best number of neighbors for our model. It will then return that ideal
    number.
    """
    y = data['num']
    X0 = data.drop('num', axis=1)
    X0 = pd.get_dummies(X0, drop_first=True)
    Xt, Xv, yt, yv = train_test_split(X0, y)
    ks = range(1, 26)
    accurt = []
    accurv = []
    for k in ks:
        m = KNeighborsClassifier(k)
        _ = m.fit(Xt, yt)
        acc = m.score(Xt, yt)
        accurt.append(acc)
        accv = m.score(Xv, yv)
        accurv.append(accv)
    _ = plt.plot(ks, accurt)
    _ = plt.plot(ks, accurv)
    maxi = max(accurv)
    index = accurv.index(maxi)
    result = print('Ideal number of Neighbors:', index)
    return result


def fit_kneighbors_to_has_disease(data: pd.DataFrame) -> confusion_matrix:
    """
    Takes heart data and creates a KNeighborsClassifier model, and
    then runs it through a range of 1 to 25 neighbors to find the
    best number of neighbors for our model. It will then take that ideal
    number and output a confusion matrix and accuracy, recall, precision,
    and f1 scores.
    """
    data['has_disease'] = (data['num'] > 0) + 0
    y = data['has_disease']
    X0 = data.drop("has_disease", axis=1)
    X0 = data.drop('num', axis=1)
    X0 = pd.get_dummies(X0, drop_first=True)
    Xt, Xv, yt, yv = train_test_split(X0, y)
    ks = range(1, 26)
    accurt = []
    accurv = []
    for k in ks:
        m = KNeighborsClassifier(k)
        _ = m.fit(Xt, yt)
        acc = m.score(Xt, yt)
        accurt.append(acc)
        accv = m.score(Xv, yv)
        accurv.append(accv)
    _ = plt.plot(ks, accurt)
    _ = plt.plot(ks, accurv)
    maxi = max(accurv)
    index = accurv.index(maxi)
    m = KNeighborsClassifier(index)
    _ = m.fit(X0, y)
    yhat = m.predict(X0)
    result = print(confusion_matrix(y, yhat), 'Accuracy:', accuracy_score(y, yhat),
                   'Recall Score:', recall_score(y, yhat), 'Precision Score',
                   precision_score(y, yhat), 'F1 Score', f1_score(y, yhat))
    return result


def main():
    # load in the data
    heart = load_in_data('heart_disease_uci.csv')
    # Remove NA values for analysis
    heart_na = remove_na(heart)
    # run KNeighbors model for all data
    print("Ideal number of neighbors, predicting num:")
    fit_kneighbors_to_num(heart_na)
    print("Confusion Matrix and Scores for all data predicting has_disease")
    fit_kneighbors_to_has_disease(heart_na)
    # Create new dataframe for only statistically significant values
    heart_new = heart[['ca', 'cp', 'oldpeak', 'num']]
    heart_new = remove_na(heart_new)
    # run KNeighbors model for statistically significant data
    print("Ideal number of neighbors for significant data, predicting num:")
    fit_kneighbors_to_num(heart_new)
    print("Confusion Matrix and Scores for significant data predicting has_disease")
    fit_kneighbors_to_has_disease(heart_new)

if __name__ == '__main__':
    main()