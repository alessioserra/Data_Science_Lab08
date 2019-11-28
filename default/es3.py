'''
Created on 28 nov 2019

@author: zierp
'''
import csv

dataset = []

with open("SummaryofWeather.csv") as file:
    for row in csv.reader(file):
        # InvoiceNo without "C"
        if "22508" == row[0]:
            temp = []
            for column in row:
                temp.append(column)
            dataset.append(temp)
