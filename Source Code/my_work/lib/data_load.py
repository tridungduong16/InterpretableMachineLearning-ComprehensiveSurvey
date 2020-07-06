import os
import numpy as np
import pandas as pd
def load_data_ihdp(path, i_exp):
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
