# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:10:17 2019

@author: rpira
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
print(pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:]))

