import numpy 
import pandas
import csv
from pandas import DataFrame
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
path_test  = 'D:/data1/NN_data/export_data_y_start_test_30.csv'
output_test = pandas.read_csv(path_test)
output_test  = output_test.to_numpy()
shape_0 = output_test.shape[0]
shape_1 = output_test.shape[1]
test_Y = output_test[:,shape_1-1]
output_test = output_test.reshape(shape_0*shape_1, 1)
scaler_test = MinMaxScaler(feature_range=(0, 1))
output_test = scaler_test.fit_transform(output_test)
output_test = output_test.reshape(shape_0,shape_1)
X_test  = output_test[0:2000,0:20]
Y_test  = output_test[0:2000,20]


# calculate predictions
loaded_model= load_model('D:/MITECH/2020/CNN/CNN_sigmoid/sigmoid_input20_60_30_1_eps6_batsz2.h5')
predictions = loaded_model.predict(X_test)
print(predictions)
predictions = scaler_test.inverse_transform(predictions)
for x in range(2000):
    print(predictions[x],'====',test_Y[x])



