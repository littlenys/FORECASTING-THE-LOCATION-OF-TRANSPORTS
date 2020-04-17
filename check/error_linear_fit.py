import numpy 
import pandas
import csv
from pandas import DataFrame
from keras.models import load_model
import array as arr
import matplotlib.pyplot as plt


path_test  = 'D:/MITECH/2020/DATA/NN_data2/test/export_data_x_center_test_40.csv'
input_x = 40
output_test = pandas.read_csv(path_test)
shape_1 = output_test.shape[1]
X_test  = output_test.iloc[:,(shape_1-1-input_x):(shape_1-1)]
Y_test  = output_test.iloc[:,shape_1-1]
shape_0 = X_test.shape[0]

loaded_model=load_model('D:/MITECH/2020/CNN/CNN_linear/linear_input_x_c_40_60_30_1_eps10_bz4.h5')
predictions = loaded_model.predict(X_test)
rounded = [round(x[0]) for x in predictions]


loss=0
max_loss = 0
min_loss = 1000
for i in range(shape_0):
    loss += abs(Y_test[i]-predictions[i])
    if (abs(Y_test[i]-predictions[i]) > max_loss and (Y_test[i] !=-1)):
        max_loss = abs(Y_test[i]-predictions[i])
        y_max_loss = i
    if abs(Y_test[i]-predictions[i]) <min_loss:
        min_loss = abs(Y_test[i]-predictions[i])
        y_min_loss = i
#print('sum loss = ',loss)
print('average loss = ',loss/shape_0)
print('max loss = ',max_loss)
print(' location = ',y_max_loss, 'value real = ',Y_test[y_max_loss], 'value predict = ', predictions[y_max_loss] )
#print('min loss = ',min_loss, ' location = ',y_min_loss, 'value real = ',Y_test[y_min_loss], 'value predict = ', predictions[y_min_loss])
'''
array=arr.array('f',)
for x in range(shape_0):
    array.append(x)

plt.plot(array,predictions,label='predict')
plt.plot(array,Y_test,label='real')
plt.legend(loc='best')
plt.show()
'''