import pandas
import numpy
import csv
import array as arr

path_x_start='D:/MITECH/2020/DATA/NN_data2/test/export_data_x_start_test_40.csv'
path_y_start='D:/MITECH/2020/DATA/NN_data2/test/export_data_y_start_test_40.csv'
path_x_end  ='D:/MITECH/2020/DATA/NN_data2/test/export_data_x_end_test_40.csv'
path_y_end  ='D:/MITECH/2020/DATA/NN_data2/test/export_data_y_end_test_40.csv'

x_start=pandas.read_csv(path_x_start)
y_start=pandas.read_csv(path_y_start)
x_end = pandas.read_csv(path_x_end)
y_end = pandas.read_csv(path_y_end)

x_start=x_start.to_numpy()
y_start=y_start.to_numpy()
x_end = x_end.to_numpy()
y_end = y_end.to_numpy()

shape_0 = x_start.shape[0]
shape_1 = x_start.shape[1]

x_center=numpy.zeros((shape_0,shape_1))
with open('D:/MITECH/2020/DATA/NN_data2/test/export_data_x_center_test_40.csv',mode='w',newline='') as file_X:
    writer_X = csv.writer(file_X, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(shape_0):
        array_X=arr.array('I',[])
        print('load',i)
        for j in range(shape_1):
            x_center[i,j] = int((int(x_start[i,j])+int(x_end[i,j]))//2)
            if (j<(shape_1-1) and (x_start[i,j]== -1 or x_end[i,j]== -1)):
                k = j
                while( k< (shape_1-1) and (x_start[i,k] ==-1 or x_end[i,k]==-1)):
                    if k < (shape_1-1):
                        k = k+1
                if( k == (shape_1-1) and ( x_start[i,k] ==-1 or x_end[i,k]==-1)):
                    while( x_start[i,k] ==-1 or x_end[i,k]==-1):
                        k = k-1
                x_center[i,k] = int((int(x_start[i,k])+int(x_end[i,k]))//2)
                array_X.append(int(x_center[i,k]))
            if  (x_start[i,j]!= -1 and x_end[i,j]!= -1) :
                    array_X.append(int(x_center[i,j]))
            if (j==(shape_1-1) and (x_start[i,j]== -1 or x_end[i,j]== -1)):
                array_X.append(int(x_center[i,j-1]))
        writer_X.writerow(array_X)

y_center=numpy.zeros((shape_0,shape_1))
with open('D:/MITECH/2020/DATA/NN_data2/test/export_data_y_center_test_40.csv',mode='w',newline='') as file_y:
    writer_y = csv.writer(file_y, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in range(shape_0):
        array_y=arr.array('I',[])
        print('load',i)
        for j in range(shape_1):
            y_center[i,j] = int((int(y_start[i,j])+int(y_end[i,j]))//2)
            if (j<(shape_1-1) and (y_start[i,j]== -1 or y_end[i,j]== -1)):
                k = j
                while( k< (shape_1-1) and (y_start[i,k] ==-1 or y_end[i,k]==-1)):
                    if k < (shape_1-1):
                        k = k+1
                if( k == (shape_1-1) and ( y_start[i,k] ==-1 or y_end[i,k]==-1)):
                    while( y_start[i,k] ==-1 or y_end[i,k]==-1):
                        k = k-1
                y_center[i,k] = int((int(y_start[i,k])+int(y_end[i,k]))//2)
                array_y.append(int(y_center[i,k]))
            if  (y_start[i,j]!= -1 and y_end[i,j]!= -1) :
                    array_y.append(int(y_center[i,j]))
            if (j==(shape_1-1) and (y_start[i,j]== -1 or y_end[i,j]== -1)):
                array_y.append(int(y_center[i,j-1]))
        writer_y.writerow(array_y)