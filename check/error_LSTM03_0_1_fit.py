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
X_test  = data_test[:,(shape_1-1-look_back):(shape_1-1)]
Y_test  = data_test[:,(shape_1-1)]
X_test = numpy.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

model=load_model('D:/MITECH/2020/LSTM/LSTM03_input_x_c_20_4_1_li_eps6_bz1.h5')
print("Loaded model from disk")

# calculate predictions
predictions = model.predict(X_test)

loss=0
max_loss = 0
min_loss = 1000
for i in range(shape_0):
    loss += abs(Y_test[i]-predictions[i])
    if (abs(Y_test[i]-predictions[i]) > max_loss and Y_test[i] != 0 ):
        max_loss = abs(Y_test[i]-predictions[i])
        y_max_loss = i
    if abs(Y_test[i]-predictions[i]) <min_loss:
        min_loss = abs(Y_test[i]-predictions[i])
        y_min_loss = i
#print('sum loss = ',loss)
print('average loss = ',loss/shape_0)
print('max loss = ',max_loss)
print(' location = ',y_max_loss, 'value real = ',Y_test[y_max_loss], 'value predict = ', predictions[y_max_loss] )

predictions = predictions.reshape(-1,1)
predictions = scaler_test.inverse_transform(predictions)
loss=0
max_loss = 0
min_loss = 1000
for i in range(shape_0):
    loss += abs(test_Y[i]-predictions[i])
    if (abs(test_Y[i]-predictions[i]) > max_loss and test_Y[i] != -1 ):
        max_loss = abs(test_Y[i]-predictions[i])
        y_max_loss = i
    if abs(test_Y[i]-predictions[i]) <min_loss:
        min_loss = abs(test_Y[i]-predictions[i])
        y_min_loss = i
#print('sum loss = ',loss)
print('average loss = ',loss/shape_0)
print('max loss = ',max_loss)
print(' location = ',y_max_loss, 'value real = ',test_Y[y_max_loss], 'value predict = ', predictions[y_max_loss] )
#print('min loss = ',min_loss, ' location = ',y_min_loss, 'value real = ',test_Y[y_min_loss], 'value predict = ', predictions[y_min_loss])
'''
array=arr.array('f',)
for x in range(shape_0):
    array.append(x)

plt.plot(array,predictions,label='predict')
plt.plot(array,test_Y,label='real')
plt.legend(loc='best')
plt.show()
'''

