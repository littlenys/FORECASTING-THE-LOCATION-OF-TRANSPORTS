import numpy 
import pandas
import csv
from pandas import DataFrame
from keras.models import load_model


path_test  = 'D:/MITECH/2020/DATA/NN_data2/test/export_data_x_center_test_40.csv'
output_test = pandas.read_csv(path_test)
X_test  = output_test.iloc[100000:120000,0:40]
Y_test  = output_test.iloc[100000:120000,40]

loaded_model=load_model('D:/MITECH/2020/CNN/CNN_linear/linear_input_x_c_40_60_30_1_eps10_bz1.h5')
predictions = loaded_model.predict(X_test)
rounded = [round(x[0]) for x in predictions]

for x in range(19999):
    print(rounded[x],'====',Y_test[x+100000])



