'''
Created on 27 nov 2019

@author: zierp
'''
import numpy as np # use numpy for vectors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestRegressor


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
    return np.sign(x)*(x**2 + 300) + 20*np.sin(x)

#Exercise 6
def inject_noise(y):
    """Add a random noise drawn from a normal distribution."""
    return y + np.random.normal(0, 50, size=y.size)

# F1
y = f(X)
""""y = inject_noise(y) #add noise, model crash down to -5%"""
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
print("Accuracy f1: ",r2_score(y_test_pred, y_test)) #Accuracy

# F2
y2 = f2(X) 
y2 = inject_noise(y2) # adding noise, model loses 30%
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, train_size=30, random_state=42, shuffle=True)
y2_test = y2_test[X2_test.argsort()]
X2_test.sort()

"""PLOT with Polynomial"""
plot.scatter(X2_test, y2_test, c ='green')
reg = make_pipeline(PolynomialFeatures(2),Lasso(alpha=0.5))
reg.fit(X2_train[:,np.newaxis], y2_train) # np.newaxis transform from ROWS VECTOR to COLUMN VECTOR
y2_test_pred = reg.predict(X2_test[:,np.newaxis])
plot.scatter(X2_test, y2_test_pred, c='blue', linewidths=0.5)
plot.show()
print("Accuracy f2: ",r2_score(y2_test_pred, y2_test)) #Accuracy

# F3
y3 = f3(X) 
y3 = inject_noise(y3) #in this way, lost less than 2%
X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3, train_size=30, random_state=42, shuffle=True)
y3_test = y3_test[X3_test.argsort()]
X3_test.sort()

"""PLOT with RandomForestRegressor"""
plot.scatter(X3_test, y3_test, c ='green')
reg = RandomForestRegressor(n_estimators=1000)
reg.fit(X3_train[:,np.newaxis], y3_train) # np.newaxis transform from ROWS VECTOR to COLUMN VECTOR
y3_test_pred = reg.predict(X3_test[:,np.newaxis])
plot.scatter(X3_test, y3_test_pred, c='blue', linewidths=0.5)
plot.show()
print("Accuracy f3: ",r2_score(y3_test_pred, y3_test)) #Accuracy