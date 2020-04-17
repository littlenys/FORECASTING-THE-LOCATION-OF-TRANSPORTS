import numpy 
import pandas
import csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Activation
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

path_train = 'D:/MITECH/2020/DATA/NN_data2/train/export_data_x_center_train_20.csv'
data_train = read_csv(path_train)
data_train = data_train.to_numpy()
shape_0 = data_train.shape[0]
shape_1 = data_train.shape[1]
# normalize the data_train
data_train = data_train.reshape(shape_0*shape_1, 1)
scaler_train = MinMaxScaler(feature_range=(0, 1))
data_train = scaler_train.fit_transform(data_train)
data_train = data_train.reshape(shape_0,shape_1)
X_train = data_train[:,0:shape_1-1]
Y_train = data_train[:,(shape_1-1)]
'''
# split into input (X) and output (Y) variables
X_train = input_train.iloc[:,0:20]
Y_train = input_train.iloc[:,20]
'''
# create modelz
model = Sequential()
model.add(Dense(60 , input_dim=20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=20, batch_size=2)

model.save('D:/MITECH/2020/CNN/CNN_linear/linear_input_x01_c_20_60_30_1_eps20_bz2.h5')
print("Saved model to disk")


