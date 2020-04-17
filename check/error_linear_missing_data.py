import numpy
import matplotlib.pyplot as plt
import array as arr
from pandas import read_csv
import math
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
path_test  = 'D:/MITECH/2020/DATA/NN_data2/test/export_data_x_center_test_40.csv'
data_test = read_csv(path_test)
data_test  = data_test.to_numpy()
shape_0 = data_test.shape[0]
shape_1 = data_test.shape[1]
test_Y = data_test[:,shape_1-1]
# normalize the data_train
look_back = 20
data_test = data_test.reshape(shape_0*shape_1, 1)
data_test = data_test.reshape(shape_0,shape_1)
X_test  = data_test[:,0:40]
Y_test  = data_test[:,40]


model=load_model('D:/MITECH/2020/CNN/CNN_linear/linear_input_x_c_20_60_30_1_eps6_bz2.h5')
print("Loaded model from disk")


predictions=numpy.zeros(shape_0)
for i in range(shape_0):
    print(i)
    test = arr.array('f',)
    print(i)
    for k in range (20):
        test.append(X_test[i,k])
    for j in range(20):
        test2 = arr.array('f',)
        test1=test
        test1=numpy.reshape(test1, (1,20 ))
        prediction_01 = model.predict(test1)
        for l in range (1,20):
            test2.append(test[l])
        test2.append(prediction_01)
        test=test2   
        prediction_02 = prediction_01[0,0]
    predictions[i] = prediction_02


loss=0
max_loss = 0
min_loss = 1000
for i in range(shape_0):
    loss += abs(test_Y[i]-predictions[i])
    if (abs(test_Y[i]-predictions[i]) > max_loss and (test_Y[i] !=-1)):
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

array=arr.array('f',)
for x in range(shape_0):
    array.append(x)
'''
plt.plot(array,predictions,label='predict')
plt.plot(array,test_Y,label='real')
plt.legend(loc='best')
plt.show()
'''