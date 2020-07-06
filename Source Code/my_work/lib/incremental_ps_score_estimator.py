import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

def influence_function(data, treatment, covariate, outcome, features, delta, model_y, model_t):

    data['p1'] = model_t.predict_proba(data[covariate])[:,1]
    data['p0'] = 1 - data['p1']

    ## Compute counterfactual outcome with no treatment
    data_pos = data.copy()
    data_pos[treatment] = 1
    data['cf1'] = model_y.predict(data_pos[features])

    ## Compute counterfactual outcome with treatment
    data_neg = data.copy()
    data_neg[treatment] = 0
    data['cf0'] = model_y.predict(data_neg[features])

    ## Compute incremental score
    data['q1'] = (delta * data['p1']) / (delta * data['p1'] + data['p0'])
    data['q1'] = data['q1'].abs()

    data['q0'] = 1 - data['q1']

    data['ips_weight'] = (data[treatment] / data['p1'] + (1 - data[treatment]) /
                          (1 - data['p1']))

    # data['ips_weight'] = np.where((data['p0'] >= data['p1']), 1 / data['p0'], 1 / data['p1'])

    data['w0'] = data['ips_weight']*data[treatment]
    data['w1'] = data['ips_weight']*(1 - data[treatment])

    data['a0'] = data['q0']*data['w0']*(data['cf0'] - data[outcome])
    data['a1'] = data['q1']*data['w1']*(data['cf1'] - data[outcome])

    influence = data['a1'] - data['a0']
    return influence

def sampling(data, treatment, covariate, outcome, delta_seq, model_t, model_y):
    features = covariate.copy()
    features.append(treatment)

    ## Compute propensity score
    data['ps_1'] = model_t.predict_proba(data[covariate])[:,1]
    data['ps_0'] = 1 - data['ps_1']

    ## Fit outcome
    data['predicted_y'] = model_y.predict(data[features])
    ## Compute counterfactual outcome with no treatment
    data_pos = data.copy()
    data_pos[treatment] = 1
    data['treated_cf_outcome'] = model_y.predict(data_pos[features])

    ## Compute counterfactual outcome with treatment
    data_neg = data.copy()
    data_neg[treatment] = 0
    data['control_cf_outcome'] = model_y.predict(data_neg[features])

    effect = []
    for delta in delta_seq:
        influence = influence_function(data, treatment, covariate, outcome, delta, model_y, model_t)
        effect.append(influence)

    return np.mean(effect, axis = 0)
