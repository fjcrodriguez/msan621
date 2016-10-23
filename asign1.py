# Assignment 1
# MSAN 621
# @author Francisco Calderon
# @date October 21, 2016

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Imputer


def mse(y, yhat, df):
    return sum( (y - yhat)**2 ) / (len(y) - df)

# ################# Boston Data #############################
boston = pd.read_csv('boston.csv')

training_y = boston.iloc[:-50, 13]
training_x = boston.iloc[:-50, :13]

test_y = boston.iloc[-50:, 13]
test_x = boston.iloc[-50:, :13]

# Linear Regression on Boston
reg = linear_model.LinearRegression()
reg.fit(training_x, training_y)
pred = reg.predict(test_x)
print 'Least Squares Regression MSE:', mse(test_y, pred, len(boston.columns)-1)

# KNN Regression on Boston
k = 10
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(training_x, training_y)
pred = knn.predict(test_x)
print 'KNN Regression MSE:', mse(test_y, pred, len(boston.columns)-1)

# ################ Weather Data ##############################

# read in and process temperature columns into integers
temp = pd.read_csv('hly-temp-normal.txt', header=None, delim_whitespace=True)
temp[3] = temp[3].str.rstrip('PCS')
for idx in range(4,27):
    temp[idx] = temp[idx].str.rstrip('PCS')

# impute missing data
col_imputer = Imputer(-9999, 'mean', 0)
for col in range(3,27):
    temp[col] = pd.to_numeric(temp[col])

temp.loc[:,3:27] = col_imputer.fit_transform(temp.loc[:,3:27])

# separate test data and training data
training = temp[~temp[0].isin(['USW00023234', 'USW00014918', 'USW00012919', 'USW00013743', 'USW00025309'])]
test = temp[temp[0].isin(['USW00023234', 'USW00014918', 'USW00012919', 'USW00013743', 'USW00025309'])]

LS_mse = []
KNN_mse = []
for i in range(3,27):
    X = pd.DataFrame()
    nIDs = len(training[0].unique())
    if i > 3:
        X['prevhour'] = training[i-1]
        X['avg_daily'] = np.mean(training.loc[:, 3:(i - 1)], axis=1)
    X['prevday'] = training.groupby([0, 1, 2])[3].mean().shift(1)
    X['avg_daily_global'] = np.repeat(training.groupby([1, 2])[i].mean(), nIDs)

    # perform least squares on test set

    # get test MSE for least squares

    # perform KNN on test set

    # get test MSE for KNN

