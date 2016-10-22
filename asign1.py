# Assignment 1
# MSAN 621
# @author Francisco Calderon
# @date October 21, 2016

import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor


def mse(y, yhat, df):
    return sum( (y - yhat)**2 ) / (len(y) - df)

################## Boston Data #############################

boston = pd.read_csv('boston.csv')

training_y = boston.iloc[:-50, 13]
training_x = boston.iloc[:-50, :13]

test_y = boston.iloc[-50:, 13]
test_x = boston.iloc[-50:, :13]

# Linear Regression on Boston
reg = linear_model.LinearRegression()
reg.fit(training_x, training_y)
pred = reg.predict(test_x)
print mse(test_y, pred, len(boston.columns)-1)

# KNN Regression on Boston
k = 10
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(training_x, training_y)
pred = knn.predict(test_x)
print mse(test_y, pred, len(boston.columns)-1)

