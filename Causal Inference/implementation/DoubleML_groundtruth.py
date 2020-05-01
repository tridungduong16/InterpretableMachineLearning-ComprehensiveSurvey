import numpy as np
from sklearn.datasets import make_spd_matrix
import math
import statsmodels.api as sm # for OLS from sklearn.ensemble import RandomForestRegressor # Our ML algorithm # Set up the environment randomseednumber = 11022018
np.random.seed(randomseednumber)
N = 500 # No. obs k=10 # = No. variables in x_i theta=0.5 # Structural parameter b= [1/k for k in range(1,11)] # x weights sigma = make_spd_matrix(k,randomseednumber) # # NUmber of simulations MC_no = 500
def g(x):
    return np.power(np.sin(x),2)
def m(x,nu=0.,gamma=1.):
    return 0.5/math.pi*(np.sinh(gamma))/(np.cosh(gamma)-np.cos(x-nu))
# Array of estimated thetas to store results theta_est = np.zeros(shape=[MC_no,3])

for i in range(MC_no):
    # Generate data: no. obs x no. variables in x_i     X = np.random.multivariate_normal(np.ones(k),sigma,size=[N,])
    G = g(np.dot(X,b))
    M = m(np.dot(X,b))
    D = M+np.random.standard_normal(size=[500,])
    Y = np.dot(theta,D)+G+np.random.standard_normal(size=[500,])
    #     # Now run the different methods     #     # OLS --------------------------------------------------     OLS = sm.OLS(Y,D)
    results = OLS.fit()
    theta_est[i][0] = results.params[0]

    # Naive double machine Learning ------------------------     naiveDMLg =RandomForestRegressor(max_depth=2)
    # Compute ghat     naiveDMLg.fit(X,Y)
    Ghat = naiveDMLg.predict(X)
    naiveDMLm =RandomForestRegressor(max_depth=2)
    naiveDMLm.fit(X,D)
    Mhat = naiveDMLm.predict(X)
    # vhat as residual     Vhat = D-Mhat
    theta_est[i][1] = np.mean(np.dot(Vhat,Y-Ghat))/np.mean(np.dot(Vhat,D))

    # Cross-fitting DML -----------------------------------     # Split the sample     I = np.random.choice(N,np.int(N/2),replace=False)
    I_C = [x for x in np.arange(N) if x not in I]
    # Ghat for both     Ghat_1 = RandomForestRegressor(max_depth=2).fit(X[I],Y[I]).predict(X[I_C])
    Ghat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],Y[I_C]).predict(X[I])
    # Mhat and vhat for both     Mhat_1 = RandomForestRegressor(max_depth=2).fit(X[I],D[I]).predict(X[I_C])
    Mhat_2 = RandomForestRegressor(max_depth=2).fit(X[I_C],D[I_C]).predict(X[I])
    Vhat_1 = D[I_C]-Mhat_1
    Vhat_2 = D[I] - Mhat_2
    theta_1 = np.mean(np.dot(Vhat_1,(Y[I_C]-Ghat_1)))/np.mean(np.dot(Vhat_1,D[I_C]))
    theta_2 = np.mean(np.dot(Vhat_2,(Y[I]-Ghat_2)))/np.mean(np.dot(Vhat_2,D[I]))
    theta_est[i][2] = 0.5*(theta_1+theta_2)