import pandas as pd
import numpy as np
import math

from dowhy import CausalModel
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

def incremenal_ps_score_estimator(df, treatment, covariate, model_t, model_y, delta = 2):
    '''
        A custom estimator based on incremental propensity score

        Parameters
        -----------
        delta: is an increment parameter
        df: dataframe
        treatment: treatment name
        covariate: confounder name
        model_t: predict treatment based on covariate X
        model_y: predict outcome based on treatment T and covariate X
    '''

    features = covariate.copy()

    features.append(treatment)

    ## Compute propensity_score
    df['ps_1'] = model_t.predict_proba(df[covariate])[:,1]
    df['ps_0'] = model_t.predict_proba(df[covariate])[:,0]

    ## Compute incremental propensity score
    df['incre_ps_1'] = (delta * df['ps_1']) / (delta * df['ps_1'] + df['ps_0'])
    df['incre_ps_0'] = 1 - df['incre_ps_1']

    ## Compute counterfactual outcome with no treatment
    df_pos = df.copy()
    df_pos['v0'] = 0
    df['treated_cf_outcome'] = model_y.predict(df_pos[features])

    ## Compute counterfactual outcome with treatment
    df_control = df.copy()
    df_control['v0'] = 1
    df['control_cf_outcome'] = model_y.predict(df_control[features])

    df['incre_effect'] = df['incre_ps_1']*df['treated_cf_outcome'] + df['incre_ps_0']*df['control_cf_outcome']

    return np.mean(df['incre_effect'])

def inlfuence_function(df, treatment, covariate, outcome, model_y, model_t):
    '''
        A custom estimator based on incremental propensity score

        Parameters
        -----------
        delta: is an increment parameter
        df: dataframe
        treatment: treatment name
        covariate: confounder name
        outcome: outcome name
        model_t: predict treatment based on covariate X
        model_y: predict outcome based on treatment T and covariate X
    '''

    features = covariate + treatment

    df['predicted_y'] = model_y.predict(df[features])

    ## Compute propensity_score
    df['ps_1'] = model_t.predict_proba(df[covariate])[:,1]
    df['ps_0'] = model_t.predict_proba(df[covariate])[:,0]

    ## Compute incremental propensity score
    df['incre_ps_1'] = (delta * df['ps_1']) / (delta * df['ps_1'] + df['ps_0'])
    df['incre_ps_0'] = 1 - df['incre_ps_1']

    ## Compute counterfactual outcome with no treatment
    df_pos = df.copy()
    df_pos['v0'] = 0
    df['treated_cf_outcome'] = rf.predict(df_pos[features])

    ## Compute counterfactual outcome with treatment
    df_control = df.copy()
    df_control['v0'] = 1
    df['control_cf_outcome'] = rf.predict(df_control[features])

    residual = df[outcome] - df['predicted_y']

    w0 = [0 if df.loc[i, 'v0'] == 1 else 1/df.loc[i, 'ps_0'] for i in range(len(df))]
    w1 = [0 if df.loc[i, 'v0'] == 0 else 1/df.loc[i, 'ps_1'] for i in range(len(df))]
    w0 = np.array(w0)
    w1 = np.array(w1)

    phi_0 = w0 * residual + df['treated_cf_outcome']
    phi_1 = w1 * residual + df['control_cf_outcome']

    m0 = df['incre_ps_1']*phi_1 + df['incre_ps_0']*phi_0
    m1 = (delta*(df['treated_cf_outcome'] - df['control_cf_outcome'])
         *(df['v0'] - df['ps_1']))/(delta*df['ps_1'] + df['ps_0'])**2

    influence = m0 + m1 - df['incre_effect']
    return influence

def sample_estimator(D0, D1, treatment, covariate, outcome, delta):
    '''
        A custom estimator based on incremental propensity score
        Parameters
        -----------
        delta: is an increment parameter
        D0: data train
        D1: data test
        treatment: treatment name
        covariate: confounder name
        outcome: outcome name
    '''
    features = covariate.copy()

    features.append(treatment)

    ## fit treatment effect on D0
    logreg = LogisticRegression()
    logreg.fit(D0[covariate], D0[treatment])

    ## predict on D1
    D1['ps_1'] = logreg.predict_proba(D1[covariate])[:,1]
    D1['ps_0'] = 1 - D1['ps_1']

    D1['incre_ps_1'] = (delta * D1['ps_1']) / (delta * D1['ps_1'] + D1['ps_0'])
    D1['incre_ps_0'] = 1 - D1['incre_ps_1']
    ## compute weight
    w_t = (delta*D1[treatment] + 1 - D1[treatment]) / (delta*D1['ps_1'] + D1['ps_0'])
    v_t = (1-delta)*(D1[treatment]*D1['ps_0']- (1-D1[treatment])*D1['ps_1']*delta) / delta

    ## fit outcome on D0
    gbr = GradientBoostingRegressor(random_state=0, n_estimators = 1000)
    gbr.fit(D0[features], D0[outcome])

    ## Compute counterfactual outcome on D1
    df_pos = D1.copy()
    df_pos[treatment[0]] = 1
    D1['treated_cf_outcome'] = gbr.predict(df_pos[features])

    df_neg = D1.copy()
    df_neg[treatment[0]] = 0
    D1['control_cf_outcome'] = gbr.predict(df_neg[features])

    m0 = np.mean(D1['treated_cf_outcome'])
    m1 = np.mean(D1['control_cf_outcome'])

    r_t = (delta*m1*D1['incre_ps_1'] + m0*D1['incre_ps_0']) / (delta*D1['incre_ps_1'] + D1['incre_ps_0'])

    effect = w_t*D1[outcome] + w_t*v_t*r_t

    del logreg
    del gbr
    del D0, D1
    del df_pos, df_neg

    return np.mean(effect)
def proposed_estimation(df_expr, treatment, covariate, outcome, delta, n_splits):
    effect = []
    for sample_delta in tqdm(delta):
        D0, D1 = train_test_split(df_expr, train_size = 0.9, random_state = 1)
        D0 = D0.reset_index().drop(columns = ['index'])
        D1 = D1.reset_index().drop(columns = ['index'])
        e = sample_estimator(D0,
                            D1,
                            treatment,
                            covariate,
                            outcome,
                            sample_delta)

        effect.append(e)
    return effect

def z_estimator(data, treatment, covariate, outcome, delta_seq, model_t, model_y):
    features = covariate.copy()
    features.append(treatment)

    ## Compute propensity score
    data['ps_1'] = model_t.predict_proba(data[covariate])[:,1]
    data['ps_0'] = 1 - data['ps_1']

    ## fit outcome
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
    for delta in tqdm(delta_seq):
        data['incre_ps_1'] = (delta * data['ps_1']) / (delta * data['ps_1'] + data['ps_0'])
        data['incre_ps_0'] = 1 - data['incre_ps_1']

        w_t = (delta*data[treatment] + 1 - data[treatment]) / (delta*data['ps_1'] + data['ps_0'])
        v_t = (data[treatment]*(data['ps_0'])
                        - (1 - data[treatment])*data['ps_1']*delta) / (delta / (1-delta))

        r_t = (data['treated_cf_outcome']*data['incre_ps_1'] + data['control_cf_outcome']*data['incre_ps_0'])

        e = w_t*data[outcome] + w_t*v_t*r_t

        residual = data[outcome] - data['predicted_y']

        w0 = [0 if data.loc[i, treatment] == 1 else 1/data.loc[i, 'ps_0'] for i in range(len(data))]
        w1 = [0 if data.loc[i, treatment] == 0 else 1/data.loc[i, 'ps_1'] for i in range(len(data))]
        w0 = np.array(w0)
        w1 = np.array(w1)

        phi_0 = w0 * residual + data['treated_cf_outcome']
        phi_1 = w1 * residual + data['control_cf_outcome']

        m0 = data['incre_ps_1']*phi_1 + data['incre_ps_0']*phi_0

        m1 = (delta*(data['treated_cf_outcome'] - data['control_cf_outcome'])
             *(data[treatment] - data['ps_1']))/(delta*data['ps_1'] + data['ps_0'])**2

        effect.append(np.mean(e))
    return np.mean(effect)
