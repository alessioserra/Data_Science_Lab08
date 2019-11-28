'''
Created on 28 nov 2019

@author: zierp
'''
import csv
import matplotlib.pyplot as plot
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics.regression import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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

# Exercise 5-6-7
"""Date from 1940 to 1944"""
data_train = []
means_train = []
for el,yy in zip(X,y):
    if np.datetime64(el) <= np.datetime64('1944-12-31'):
        data_train.append(el)
        means_train.append(yy)

"""Date in 1945"""
data_test = []
means_test = []
for el,yy in zip(X,y):
    if np.datetime64(el) > np.datetime64('1944-12-31'):
        data_test.append(el)
        means_test.append(yy)

"""WINDOW"""
l_train = len(data_train)
windows_train = []
y_train = []
W=9 # suggested by Nick
days_to_predict=2

"""Windows train"""
for i in range(l_train-2*W-days_to_predict):
    windows_train.append(means_train[i:i+W+1])
    y_train.append(means_train[i+W+days_to_predict])

l_test = len(data_test)
windows_test = []
y_test = []

"""Windows test"""
for i in range(l_test-2*W-days_to_predict):
    windows_test.append(means_test[i:i+W+1])
    y_test.append(means_test[i+W+days_to_predict])
   
"""Convert type"""
windows_train = np.array(windows_train).astype('float32')
windows_test = np.array(windows_test).astype('float32')
y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')
    
"""Random Forest"""
reg = RandomForestRegressor(1000,n_jobs=-1)
reg.fit(windows_train,y_train)
y_pred = reg.predict(windows_test)
print('R2 score of RandomForestRegressor:',r2_score(y_test,y_pred))

"""Lasso"""
reg2 = make_pipeline(PolynomialFeatures(5), Lasso(alpha=0.5,fit_intercept=True,max_iter=1000))
reg2.fit(windows_train,y_train)
y_pred2 = reg2.predict(windows_test)
print('R2 score of Lasso:',r2_score(y_test,y_pred2))

"""Ridge"""
reg3 = make_pipeline(PolynomialFeatures(3), Ridge(alpha=0.5,fit_intercept=True))
reg3.fit(windows_train,y_train)
y_pred3 = reg3.predict(windows_test)
print('R2 score of Ridge:',r2_score(y_test,y_pred3))

"""SVR"""
reg4 = SVR(gamma='scale')
reg4.fit(windows_train,y_train)
y_pred4 = reg4.predict(windows_test)
print('R2 score of SVR:',r2_score(y_test,y_pred4))


"""Final plot"""
if plot:
    fig = plot.figure()  
    plot.plot(np.arange(0,len(y_test)), y_test, c='red')
    #plt.plot(np.arange(0,len(y_test)), y_pred2, c='red')
    plot.plot(np.arange(0,len(y_test)), y_pred4, c='blue')
    plot.show()
