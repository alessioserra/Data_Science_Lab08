'''
Created on 27 nov 2019

@author: zierp
'''
import numpy as np # use numpy for vectors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import r2_score

tr = 20
n_samples = 100
X = np.linspace(-tr, tr, n_samples)

"""f1(x) = x * sin(x) + 2x""" 
def f(x):
    return x * np.sin(x) + 2 * x

"""f2(x) = 10 sin(x) + x^2"""
def f2(x):
    return 10 * np.sin(x) + x**2

"""f3(x) = sign(x)(x^2 + 300) + 20 sin(x) """
def f3(x):
    return 0

# Es1
y = f(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=30, random_state=42, shuffle=True)
y_test = y_test[X_test.argsort()]
X_test.sort()

"""PLOT with Linear Regressor"""
plot.scatter(X_test, y_test, c ='green')
reg = LinearRegression(fit_intercept = True)
reg.fit(X_train[:,np.newaxis], y_train) # np.newaxis transform from ROWS VECTOR to COLUMN VECTOR
y_test_pred = reg.predict(X_test[:,np.newaxis])
plot.scatter(X_test, y_test_pred, c='blue', linewidths=0.5)
plot.show()
print(r2_score(y_test_pred, y_test)) #Accuracy


