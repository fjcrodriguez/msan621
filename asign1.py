# Assignment 1
# MSAN 621
# @author Francisco Calderon
# @date October 21, 2016

import pandas as pd
from sklearn import linear_model

def mse(y, yhat, df):
    return sum( (y - yhat)**2 ) / (len(y) - df)

boston = pd.read_csv('boston.csv')

training_y = boston.iloc[:-50, 13]
training_x = boston.iloc[:-50, :13]

test_y = boston.iloc[-50:, 13]
test_x = boston.iloc[-50:, :13]

reg = linear_model.LinearRegression()
reg.fit(training_x, training_y)
pred = reg.predict(test_x)
print mse(test_y, pred, len(boston.columns)-1)

