# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:42:16 2019

@author: rpira
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata
import numpy as np 
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
m, n = housing.data.shape
#j= housing.target.shape

#from keras.datasets import mnist

####### Data Minst

#mnist = fetch_mldata('MNIST original')
#######  X, Y
#X, y = mnist["data"], mnist["target"]
#X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
######
#X_train=housing.data[1]
#y_train=housing.data[1]
#X_test=housing.data[1]

X_train=[1,2,3]
y_train=[2,3,4]
X_test=[1,2,3]

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train) 
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10, feature_columns=feature_columns)
#dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)

#y_pred = list(dnn_clf.predict(X_test))
#accuracy_score(y_test, y_pred)
