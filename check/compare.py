import matplotlib.pyplot as plt
import array as arr
import numpy
from pandas import read_csv
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
start=159131
end=159230
look_back = 20
path_test  = 'D:/MITECH/2020/DATA/NN_data2/test/export_data_x_center_test_20.csv'
data_test = read_csv(path_test)
data_test  = data_test.to_numpy()
shape_0 = data_test.shape[0]
shape_1 = data_test.shape[1]
test_X = data_test[start:end,(shape_1-1-look_back):(shape_1-1)]
test_Y= data_test[start:end,shape_1-1]


# test for linear
test_X1 = test_X

#_____________________________________________________________________________
data_test = data_test.reshape(shape_0*shape_1, 1)
#____________________________________________________________________________
#test for LSTM01
scaler_test2 = MinMaxScaler(feature_range=(0, 1))
data_test2 = scaler_test2.fit_transform(data_test)
data_test2 = data_test2.reshape(shape_0,shape_1)
test_X2 = data_test2[start:end,(shape_1-1-look_back):(shape_1-1)]
test_X2 = numpy.reshape(test_X2, (test_X2.shape[0], 1, test_X2.shape[1]))

#test for LSTM03 1
scaler_test3 = MinMaxScaler(feature_range=(0, 1))
data_test3 = scaler_test3.fit_transform(data_test)
data_test3 = data_test3.reshape(shape_0,shape_1)
test_X3 = data_test3[start:end,(shape_1-1-look_back):(shape_1-1)]
test_X3 = numpy.reshape(test_X3, (test_X3.shape[0],test_X3.shape[1],1))

#test for LSTM03 1
scaler_test4 = MinMaxScaler(feature_range=(0, 1))
data_test4 = scaler_test4.fit_transform(data_test)
data_test4 = data_test4.reshape(shape_0,shape_1)
test_X4 = data_test4[start:end,(shape_1-1-look_back):(shape_1-1)]
test_X4 = numpy.reshape(test_X4, (test_X4.shape[0],test_X4.shape[1],1))

#test for LSTM02
model1=load_model('D:/MITECH/2020/CNN/CNN_linear/linear_input_x01_c_20_60_30_1_eps20_bz2.h5')
model2=load_model('D:/MITECH/2020/LSTM/LSTM01_input_x_c_20_40_1_eps100_bz32.h5')
model3=load_model('D:/MITECH/2020/LSTM/LSTM03_input_x_c_20_2_1_li_eps4_bz1.h5')
model4=load_model('D:/MITECH/2020/LSTM/LSTM03_input_x_c_20_4_1_li_eps6_bz1.h5')

predictions1 = model1.predict(test_X1)
predictions2 = model2.predict(test_X2)
predictions3 = model3.predict(test_X3)
predictions4 = model4.predict(test_X4)

predictions2 = scaler_test2.inverse_transform(predictions2)
predictions3 = scaler_test2.inverse_transform(predictions3)
predictions4 = scaler_test2.inverse_transform(predictions4)
array_=arr.array('f',)
for x in range(start,end):
    array_.append(x)


plt.title("test on export_data_x_center_test_20.csv")
plt.plot(array_,predictions1,label='linear_input_x01_c_20_60_30_1_eps20_bz2')
plt.plot(array_,predictions2,label='LSTM01_input_x_c_20_40_1_eps100_bz32')
plt.plot(array_,predictions3,label='LSTM03_input_x_c_20_2_1_li_eps4_bz1')
plt.plot(array_,test_Y,label='real') # red
plt.plot(array_,predictions4,label='LSTM03_input_x_c_20_4_1_li_eps6_bz1')

plt.legend(loc='best')
plt.show()
