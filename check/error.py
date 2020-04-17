import numpy
import array as arr
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
path_test  = 'D:/data1/NN_data/export_data_x_start_test_30.csv'
data_test = read_csv(path_test)
data_test  = data_test.to_numpy()
shape_0 = data_test.shape[0]
shape_1 = data_test.shape[1]
Y_test = data_test[:,shape_1-1]
# normalize the data_train
look_back = 20
data_test = data_test.reshape(shape_0*shape_1, 1)
scaler_test = MinMaxScaler(feature_range=(0, 1))
data_test = scaler_test.fit_transform(data_test)
data_test = data_test.reshape(shape_0,shape_1)
testX  = data_test[0:3000,(shape_1-1-20):(shape_1-1)]
testY  = data_test[0:3000,(shape_1-1)]

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model=load_model('D:/MITECH/2020/LSTM/LSTM_input_x_20_20_1_eps6_batsz1.h5')
print("Loaded model from disk")

# calculate predictions
predictions = model.predict(testX)
predictions = scaler_test.inverse_transform(predictions)

loss=0
max_loss = 0
min_loss = 1000
for i in range(shape_0):
    loss += abs(Y_test[i]-predictions[i])
    if (abs(Y_test[i]-predictions[i]) > max_loss and abs(Y_test[i]-predictions[i]) ):
        max_loss = abs(Y_test[i]-predictions[i])
        y_max_loss = i
    if abs(Y_test[i]-predictions[i]) <min_loss:
        min_loss = abs(Y_test[i]-predictions[i])
        y_min_loss = i
print('sum loss = ',loss)
print('average loss = ',loss/shape_0)
print('max loss = ',max_loss, ' location = ',y_max_loss)
print('min loss = ',min_loss, ' location = ',y_min_loss)

array=arr.array('f',)
for x in range(shape_0):
    array.append(x)

plt.plot(array,predictions,label='predict')
plt.plot(array,Y_test,label='real')
plt.legend(loc='best')
plt.show()
