# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:30:19 2019

@author: rpira
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
import pandas as pd

n_inputs = 7
n_hidden1 = 3
n_hidden2 = 2 
n_outputs = 1

df = pd.read_csv('data2.csv') # read data set using pandas



X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X") 
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"): 
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1") 
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2") 
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"): 
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    
learning_rate = 0.01
with tf.name_scope("train"): 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"): 
    correct = tf.nn.in_top_k(logits, y, 1) 
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


