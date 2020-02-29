# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:06:49 2019

@author: Alexander Hurley

ML 1 - Polynomial

Dataset divorce.csv was taken from https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
and modified by Alexander Hurley for use in the INFR 3700U final project. 

"""

# Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

# Reading data from file to the "divorce" variable, seperating the data where any number of spaces appear
divorce = pd.read_csv("divorce.csv",
                      skiprows = 1,
                      names=["atr1", "atr2", "atr3", "atr4", "atr5", "atr6", "atr7", "atr8", "atr9",
                             "atr10", "atr11", "atr12", "atr13", "atr14", "atr15", "atr16", "atr17",
                             "atr18", "atr19", "atr20", "atr21", "atr22", "atr23", "atr24", "atr25",
                             "atr26", "atr27", "atr28", "atr29", "atr30", "atr31", "atr32", "atr33",
                             "atr34", "atr35", "atr36", "atr37", "atr38", "atr39", "atr40", "atr41", 
                             "atr42", "atr43", "atr44", "atr45", "atr46", "atr47", "atr48", "atr49", 
                             "atr50", "atr51", "atr52", "atr53", "atr54", "divorce"])

#print(divorce)

# Splitting our training and testing data in an 80/20 split
from sklearn.model_selection import train_test_split
train, test = train_test_split(divorce, test_size=0.2)
train_predict = train.iloc[:,-1]
train_divorce = train.drop(["divorce"], axis=1)
test_predict = test.iloc[:,-1]
test_divorce = test.drop(["divorce"], axis=1)

# Building train and test data numpy array
train_divorce = np.c_[train_divorce]
train_predict = np.c_[train_predict]
test_divorce = np.c_[test_divorce]
test_predict = np.c_[test_predict]

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=1, include_bias=False)
polyX = poly_features.fit_transform(train_divorce)
linReg = linear_model.LinearRegression()
linReg.fit(polyX, train_predict)
polyPred = poly_features.transform(test_divorce)
pred = linReg.predict(polyPred)

# Calculate MSE
polyMSE = mse(test_predict, pred)
print("Polynomial Model MSE: ", polyMSE)