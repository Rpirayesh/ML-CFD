# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:21:56 2019

@author: rpira
"""

import numpy as np 
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing() 
m, n = housing.data.shape 
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X") 
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y") 
XT = tf.transpose(X) 
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()