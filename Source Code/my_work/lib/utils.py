import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_data(path, i_exp):
    col =  ["t", "yf", "ycf", "mu0", "mu1" ]
    cov = ["x" + str(i) for i in range(1,26)]
    col = col + cov
    features = cov + ["t"]
    D = np.load(path)
    df = pd.DataFrame(columns=col)

    for i in range(1,26):
        df['x' + str(i)]  = D['x'][:,i-1,i_exp-1]

    df['t']  = D['t'][:,i_exp-1:i_exp]
    df['yf'] = D['yf'][:,i_exp-1:i_exp]
    df['ycf'] = D['ycf'][:,i_exp-1:i_exp]
    df['mu0'] = D['mu0'][:,i_exp-1:i_exp]
    df['mu1'] = D['mu1'][:,i_exp-1:i_exp]
    return df


def rmse_ite(influence, true_ite):
    return np.sqrt(np.mean(np.square(true_ite - influence)))

def abs_ate(true_effect, estimation_effect):
    return np.abs(true_effect - estimation_effect)

def pehe(mu1, mu0, ypred1, ypred0):
    return np.sqrt(np.mean(np.square((mu1 - mu0) - (ypred1 - ypred0))))
