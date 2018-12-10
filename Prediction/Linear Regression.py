import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Data Preprocessing

#2.1. Data read
veriler = pd.read_csv('sales.csv')

#Data preprocessing,seperation of each columns for test and train
aylar = veriler[['Months']]
#test
print(aylar) #independent variable
satislar = veriler[['Sales']]
print(satislar) #dependent variable

#Division of the data as test and train
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0) #train and test variables defined
#model construction(Linear Regression)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train) #train objects received to be trained
tahmin=lr.predict(x_test)
x_train=x_train.sort_index() 
y_train=y_train.sort_index() # sorted to get rid of randomized elements
plt.plot(x_train,y_train) #data visualization
plt.plot(x_test,lr.predict(x_test)) #see the regression,drawing line
plt.title("Sales according to months") #Title
plt.xlabel("Months") 
plt.ylabel("Sales") # labels for axis(unnecessary)





