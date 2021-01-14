#Install the dependencies
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
plt.style.use('bmh')

df = pd.read_csv('/Users/kirankumar/Desktop/EOD-AAPL.csv')
df.head(6)

plt.figure(figsize=(16,8))
plt.title('Microsoft', fontsize = 18)
plt.xlabel('Days', fontsize= 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(df['Close'])
#plt.show()

df = df[['Close','Open','High','Low','Volume','Adj_Low','Adj_High','Adj_Open','Adj_Volume']]
df.head(4)


#we want to predict in the next 10 days, make a variable
future_day = 25
#make a column for the predicted values
df['Predictions'] = df[['Close']].shift(-future_day)
#print the data



#create feature data set and print it(missing last 25)
x = np.array(df.drop(['Predictions'], 1))[:-future_day]




#create target data set(missing first 25)
y = np.array(df['Predictions'])[:-future_day]






#split the data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)



#we are going to train a Ridge
#Tracer()()
Rid = Ridge().fit(x_train, y_train)

#get the featured data
x_future = df.drop(['Predictions'], 1)[:-future_day]
x_future = x_future.tail(future_day)





#convert
x_future = np.array(x_future)
x_future


#prediction
Rid_prediction = Rid.predict(x_future)
print(Rid_prediction)


#y_future
y_future = (df['Predictions'])[:-future_day]
y_future = y_future.tail(future_day)
y_future = np.array(y_future)

#mse
#Tracer()()
#compare the mse with other trials where we increase the number of features
#Rid_prediction=np.expand_dims(Rid_prediction,axis=1)
mse = np.mean((Rid_prediction - y_future)**2)
print('\nMean Sqaured Error = ',mse )



correlation_matrix = np.corrcoef(Rid_prediction, y_future)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2.
print(r_squared)
