import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from IPython.core.debugger import Tracer


df = pd.read_csv('/Users/kirankumar/Desktop/EOD-AAPL.csv')

#create a data frame with only closing
data = df.filter(['Close'])


#convert into an numpy array,np is a grid of values that are tupled by nonnegative integers
dataset = data.values
print(len(dataset))

# if you want the length of a specific amount
training_data_len = math.ceil( len(dataset) *.8)
print(training_data_len)

#we want to scale the data,for better optimization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data= scaler.fit_transform(dataset)


#we are creating scaled data
train_data = scaled_data[0:training_data_len , :]



#split the data into two sets
x_train =[]
y_train =[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
#we want to convert into numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

valid_data = x_train[int(len(x_train)*0.67):,:]
x_train=x_train[:int(len(x_train)*0.67),:]

valid_label = y_train[int(len(y_train)*0.67):]
y_train = y_train[:int(len(y_train)*0.67)]


#lstm require a 3D input
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
valid_data = np.reshape(valid_data,(valid_data.shape[0],valid_data.shape[1],1))

#Build
model = Sequential()
#the return sequence is the gate that lets in info from the last output
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

#complie
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#train the model
model.fit(x_train,y_train,validation_data=(valid_data,valid_label), epochs=10,batch_size=32)

#we are now going to test with actual values
test_data = dataset[training_data_len-60:,:]
print(test_data)
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
#convert x_test into numpy array
x_test = np.array(x_test)
#3D input

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#predictions
predictions = model.predict(x_test)
Tracer()()
predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse



