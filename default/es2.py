'''
Created on 28 nov 2019

@author: zierp
'''
from sklearn.datasets.samples_generator import make_regression
import numpy as np # use numpy for vectors
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def inject_noise(y):
    """Add a random noise drawn from a normal distribution."""
    return y + np.random.normal(0, 50, size=y.size)

# Exercise1
X, y = make_regression(n_samples=2000, random_state=42)
reg = RandomForestRegressor(n_estimators=10)
r2 = np.mean(cross_val_score(reg, X, y, cv=5, scoring='r2'))
print("Accuracy NO-noise : ",r2)

# Exercise 2
X2, y2 = make_regression(n_samples=2000, random_state=42)
y2 = inject_noise(y2)
reg2 = RandomForestRegressor(n_estimators=10)
r22 = np.mean(cross_val_score(reg2, X2, y2, cv=5, scoring='r2'))
print("Accuracy with-Noise: ",r22)
