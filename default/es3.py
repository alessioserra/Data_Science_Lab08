'''
Created on 28 nov 2019

@author: zierp
'''
import csv
import matplotlib.pyplot as plot
import numpy as np
from IPython.utils.py3compat import xrange
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics.classification import classification_report
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import r2_score

def rolling_window(seq, window_size):
    it = iter(seq)
    win = [it.next() for cnt in xrange(window_size)] # First window
    yield win
    for e in it: # Subsequent windows
        win[:-1] = win[1:]
        win[-1] = e
        yield win

dataset = []

with open("SummaryofWeather.csv") as file:
    for row in csv.reader(file):
        # InvoiceNo without "C"
        if "22508" == row[0]:
            temp = []
            for column in row:
                temp.append(column)
            dataset.append(temp)
            
dataset = np.array(dataset)

""" Date format and type conversion by Nick"""
for idx,row in enumerate(dataset):   
    date = row[1].split('-')
    month = date[1]
    day = date[2]
    if len(date[1]) == 1:
        month = '0'+str(date[1])
    if len(date[2]) == 1:
        day = '0'+str(date[2])
    new_date = date[0]+'-'+month+'-'+day
    dataset[idx][1] = np.datetime64(new_date)
    
"""Populate X and y"""
X = []
y = []
for row in dataset:
    X.append(row[1])
    y.append(float(row[6]))
    
X_date = []
for el in X:
    X_date.append(np.datetime64(el))
X = np.array(X)
X_date = np.array(X_date)
y = np.array(y)
y = y[X.argsort()]
X_date.sort()

"""PLOT"""
plot.plot(X_date, y, c='green')
plot.legend("Mean temp")
plot.title("Sensor 22508")
plot.xlabel("Date")
plot.ylabel("Temperature Celsius")
plot.show()

# Exercise 5
T = len(X)
W=9 # suggested by Nick
forecast = []
days_to_predict=3
"""..."""


# Exercise 6
"""Date from 1940 to 1944"""
data_train = []
y_train = []
for el,yy in zip(X,y):
    if np.datetime64(el) <= np.datetime64('1944-12-31'):
        data_train.append(el)
        y_train.append(yy)
    

"""Date in 1945"""
data_test = []
y_test = []
for el,yy in zip(X,y):
    if np.datetime64(el) > np.datetime64('1944-12-31'):
        data_test.append(el)
        y_test.append(yy)

"""Convert date"""
d_train = []
for i in enumerate(data_train):
    d_train.append(i[0])
    
d_test = []
for i in enumerate(data_test):
    d_test.append(i[0]) 
    
d_train = np.array(d_train)
d_test = np.array(d_test)

reg = RandomForestRegressor(n_estimators=100)
reg.fit(d_train.reshape(-1,1), y_train)
y_pred = reg.predict(d_test.reshape(-1,1))
print("Accuracy : ",r2_score(y_pred, y_test)) #Accuracy