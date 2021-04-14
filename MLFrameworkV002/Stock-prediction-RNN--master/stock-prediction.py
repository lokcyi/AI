#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:50:17 2018

@author: likono
"""
########### Data Preprocessing ############

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import training set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#creating a numpy array upper bound is excluded so 1:2 is same as taking the 2nd index only
training_set = dataset_train.iloc[:, 1:2].values

#Feature Scaling Using normalization Xnorm = x-min(x)/max(x)-min(x)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
#rnn will learn from 60(3 months) timesteps of previous data and try producing an output
#x_train => input(60 previous financial inputs) y_train(the next day financial results) => output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#########Building RNN#############

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# initialising the RNN
regressor = Sequential()

#Adding first LSTM and Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#LSTM2
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#LSTM3
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#LSTM4
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#output layer
regressor.add(Dense(units = 1))


#compiling RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()  







