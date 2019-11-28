'''
Created on 28 nov 2019

@author: zierp
'''
import csv
import matplotlib.pyplot as plot
import numpy as np

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
    X.append(np.datetime64(row[1]))
    y.append(float(row[6]))
X = np.array(X)
y = np.array(y)
y = y[X.argsort()]
X.sort()

"""PLOT"""
plot.scatter(X, y, c='blue', linewidths=0.1)
plot.title("Sensor 22508")
plot.xlabel("Date")
plot.ylabel("Temperature Celsius")
plot.show()