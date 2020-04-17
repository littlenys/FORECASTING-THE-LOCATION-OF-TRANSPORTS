import numpy 
import pandas
import csv
import os
import tensorflow as tf
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense , Dropout, Activation
from keras.models import model_from_json
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # use GPU device, CUDA_VISIBLE_DEVICES=-1 => use CPU
path_train = 'D:/data1/NN_data/export_data_x_start_train_40.csv'
input_train = pandas.read_csv(path_train)
scaler = MinMaxScaler(feature_range=(0, 1))
input_train = scaler.fit_transform(input_train)
X_train = input_train[:,10:30]
Y_train = input_train[:,30]
# create modelz
model = Sequential()
model.add(Dense(60, input_dim=20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=6, batch_size=2)
model.save('D:/MITECH/2020/CNN/CNN_sigmoid/sigmoid_input20_60_30_1_eps6_batsz2.h5')
print("Saved model to disk")




