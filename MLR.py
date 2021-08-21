# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 18:56:00 2021

@author: Imam Qazi
"""
# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


#encoding the character values into numbers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
x[:, 3] = labelencoder.fit_transform(x[:,3])
onehotencoder = ColumnTransformer(
    [('OHE', OneHotEncoder(),[3])],     remainder = 'passthrough'
    )
x = onehotencoder.fit_transform(x)

#test train spliting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state = 0)

#fiting MLR to the training set
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train)

#predicting or testing MLR model
prediction = mlr.predict(x_test)

