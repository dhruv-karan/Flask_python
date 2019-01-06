# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 21:14:26 2018

@author: dhruv
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('weight-height.csv')

dataset.info()
dataset.describe()

X = dataset.iloc[:,1:3].values
y = dataset.iloc[:,0:1].values


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
y[:, 0] = labelencoder.fit_transform(y[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
# Avoiding the Dummy Variable Trap
y = y[:, 1:]
# 1 represent Male

#making traing and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)



# question1: pred male or female on given data set.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred  = y_pred>0.5
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# this model workd with 93% accuracey
# tommorow we will preform another method over it so that we can increase accuracye



#3333333333333333 pickling
import pickle

pickle.dump(regressor, open("model.pkl","wb"))

#loading a model from a file called model.p




