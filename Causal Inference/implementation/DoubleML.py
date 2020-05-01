#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 07:06:48 2020

@author: dtd
"""
import numpy as np
from sklearn.datasets import make_spd_matrix
import math
import matplotlib.pyplot as plt
from statsmodels.api import OLS
from sklearn.ensemble import RandomForestRegressor

randomseednumber = 1000
sigma = make_spd_matrix(k,randomseednumber) 
b= [1/k for k in range(1,11)]


np.random.seed(100)


N = 500
k = 10
theta = 0.5
sigma = make_spd_matrix(k,randomseednumber) # 

def g(x):
    return np.power(np.sin(x),2)

def m(x, nu = 0, gamma = 1):
    return (0.5 * math.pi)*np.sinh(gamma)/(np.cosh(gamma) - np.cos(x-nu))

###Create data 
X = np.random.multivariate_normal(np.ones(k),sigma,size=[N,])
U = np.random.standard_normal(size = [500,])
V = np.random.standard_normal(size = [500,])
Y = np.dot(theta, D) + g(np.dot(X,b)) + U
D = m(np.dot(X,b)) + V
OLS_model = OLS(Y,D)
result = OLS_model.fit()

###Naive double machine learning
naiveMl1 = RandomForestRegressor() # X -> Y
naiveMl1.fit(X,Y)
Vhat1 = Y - naiveMl1.predict(X)

naiveMl2 = RandomForestRegressor() # X -> Y
naiveMl2.fit(X,D)
Vhat2 = D - naiveMl1.predict(X)

np.mean(np.dot(Vhat1, Vhat2)) / np.mean(np.dot(Vhat2, D))

