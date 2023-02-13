import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# Read in the data
#a = pd.read_csv("CFD_data.csv", delimiter=",")
b = pd.read_csv("CFDAnalasis.csv", delimiter=",")

# Extract the labels and inputs
column_names_Input = ['Alpha', 'W', 'Dh', 'L', 'L/D', 'Pe']
column_names_Label = [ 'Nu']
#Input = pd.read_csv("CFD_data.csv", delimiter=",")
#labels = pd.read_csv("CFD_data.csv", delimiter=",")

Input = pd.read_csv("CFD_data.csv", delimiter=",", usecols=column_names_Input)
labels = pd.read_csv("CFD_data.csv", delimiter=",", usecols=column_names_Label)




Xtest = b

# Train the network
X_train, X_test, y_train, y_test = train_test_split(Input, labels, test_size=0.3, random_state=42)
model = Sequential()
model.add(Dense(10, input_dim=Input.shape[1], activation='relu'))
model.add(Dense(5, input_dim=Input.shape[1], activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Test the network
y_pred = model.predict(X_test)
pre_mape = np.abs((y_pred - y_test) / y_test)
mape = np.mean(pre_mape)
Predicted= model.predict(Xtest)

print("MAPE:", mape)
print("MAPE:", Predicted)
