import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset,look_back):
    dataX,dataY = [],[]
    for j in range(len(dataset)-look_back-1):
        a = dataset[j:(j+look_back)]
        dataX.append(a)
        dataY.append(dataset[j+look_back])
    return numpy.array(dataX) , numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
path_train = 'D:/MITECH/2020/DATA/NN_data2/train/export_data_x_center_train_40.csv'
data_train = read_csv(path_train)
data_train = data_train.to_numpy()
shape_0 = data_train.shape[0]
shape_1 = data_train.shape[1]
# normalize the data_train
look_back = 20
data_train = data_train.reshape(shape_0*shape_1, 1)
scaler_train = MinMaxScaler(feature_range=(0, 1))
data_train = scaler_train.fit_transform(data_train)
data_train = data_train.reshape(shape_0,shape_1)



model = Sequential()
model.add(LSTM(20, input_shape=(1, look_back)))
model.add(Dense(1,))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(shape_0):
    dataset_train = data_train[i,:]
    trainX,trainY = create_dataset(dataset_train,20) 
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    model.fit(trainX, trainY, epochs=5, batch_size=20, verbose=1)
    if i//100000 == 0:
        print(i)
model.save('D:/MITECH/2020/LSTM/LSTM_missing_data_input_x_c_20_20_1_eps5_batsz20.h5')







