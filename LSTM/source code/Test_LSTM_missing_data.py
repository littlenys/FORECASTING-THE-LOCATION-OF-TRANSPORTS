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
start = 0
end = 100
data_test = data_test.reshape(shape_0*shape_1, 1)

scaler_test = MinMaxScaler(feature_range=(0, 1))
data_test = scaler_test.fit_transform(data_test)
data_test = data_test.reshape(shape_0,shape_1)
testX  = data_test[start:end,0:40]
testY  = data_test[start:end,40]


model=load_model('D:/MITECH/2020/LSTM/LSTM_missing_data_input_x_c_20_20_1_eps5_batsz20.h5')
print("Loaded model from disk")


predictions=numpy.zeros(end-start)
for i in range(end-start):
    test = arr.array('f',)
    print(i)
    for k in range (20):
        test.append(testX[i,k])
    for j in range(20):
        test2 = arr.array('f',)
        test1=test
        test1=numpy.reshape(test1, (1, 1, 20 ))
        prediction_01 = model.predict(test1)
        for l in range (1,20):
            test2.append(test[l])
        test2.append(prediction_01)
        print(prediction_01 , '====' , testX[i,20+j])
        test=test2   
        prediction_02 = prediction_01[0,0]
    predictions[i] = prediction_02

print(predictions)
# calculate predictions
predictions = predictions.reshape(start-end,1)
predictions = scaler_test.inverse_transform(predictions)
#rounded = [round(x[0]) for x in predictions]
for x in range(end-start):
    print(predictions[x],'====',test_Y[x])