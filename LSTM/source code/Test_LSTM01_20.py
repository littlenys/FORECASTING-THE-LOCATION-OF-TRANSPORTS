import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
path_test  = 'D:/MITECH/2020/DATA/NN_data2/test/export_data_y_center_test_20.csv'
data_test = read_csv(path_test)
data_test  = data_test.to_numpy()
shape_0 = data_test.shape[0]
shape_1 = data_test.shape[1]
test_Y = data_test[:,shape_1-1]
# normalize the data_train
look_back = 20
data_test = data_test.reshape(shape_0*shape_1, 1)
scaler_test = MinMaxScaler(feature_range=(0, 1))
data_test = scaler_test.fit_transform(data_test)
data_test = data_test.reshape(shape_0,shape_1)
testX  = data_test[0:3000,(shape_1-1-20):(shape_1-1)]
testY  = data_test[0:3000,(shape_1-1)]

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model=load_model('D:/MITECH/2020/LSTM/LSTM_input_x_c_20_20_1_eps6_bz1.h5')
print("Loaded model from disk")

# calculate predictions
predictions = model.predict(testX)
predictions = scaler_test.inverse_transform(predictions)
rounded = [round(x[0]) for x in predictions]

for x in range(3000):
    print(rounded[x],'====',test_Y[x])



