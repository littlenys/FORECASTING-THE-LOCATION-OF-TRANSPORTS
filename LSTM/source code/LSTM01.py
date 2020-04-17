import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential,load_model,save_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
path_train = 'D:/MITECH/2020/DATA/NN_data2/train/export_data_x_center_train_40.csv'
data_train = read_csv(path_train)
data_train = data_train.to_numpy()
shape_0 = data_train.shape[0]
shape_1 = data_train.shape[1]
# normalize the data_train
look_back = 40
data_train = data_train.reshape(shape_0*shape_1, 1)
scaler_train = MinMaxScaler(feature_range=(0, 1))
data_train = scaler_train.fit_transform(data_train)
data_train = data_train.reshape(shape_0,shape_1)
X_train = data_train[:,0:(shape_1-1)]
Y_train = data_train[:,(shape_1-1)]

X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back)))
model.add(Dense(1,))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=200, batch_size=32)
'''
path_train = 'D:/data1/NN_data/export_data_x_start_train_30.csv'
data_train = read_csv(path_train)
data_train = data_train.to_numpy()
shape_0 = data_train.shape[0]
shape_1 = data_train.shape[1]
# normalize the data_train
look_back = 30
data_train = data_train.reshape(shape_0*shape_1, 1)
scaler_train = MinMaxScaler(feature_range=(0, 1))
data_train = scaler_train.fit_transform(data_train)
data_train = data_train.reshape(shape_0,shape_1)
X_train = data_train[:,0:(shape_1-1)]
Y_train = data_train[:,(shape_1-1)]

X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
model.fit(X_train, Y_train, epochs=6, batch_size=2, verbose=1)
'''

model.save('D:/MITECH/2020/LSTM/LSTM_input_x_c_40_20_1_eps200_bz32.h5')







