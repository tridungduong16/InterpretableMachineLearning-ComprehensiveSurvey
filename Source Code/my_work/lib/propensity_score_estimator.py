import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors

def propensity_score_estimator_k_nearest_neighbor(df, treatment, outcome, propensity_score):
    treated = df.loc[df[treatment] == 1]
    control = df.loc[df[treatment] == 0]


    control_neighbors = (
        NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        .fit(control[propensity_score].values.reshape(-1, 1))
    )
    distances, indices = control_neighbors.kneighbors(treated[propensity_score].values.reshape(-1, 1))

    att = 0
    numtreatedunits = treated.shape[0]
    for i in range(numtreatedunits):
        treated_outcome = treated.iloc[i][outcome].item()
        control_outcome = control.iloc[indices[i]][outcome].item()
        att += treated_outcome - control_outcome

    att /= numtreatedunits

    control_neighbors = (
        NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        .fit(treated[propensity_score].values.reshape(-1, 1))
    )
    distances, indices = control_neighbors.kneighbors(control[propensity_score].values.reshape(-1, 1))

    atc = 0
    numcontrolunits = control.shape[0]
    for i in range(numcontrolunits):
        control_outcome = control.iloc[i][outcome].item()
        treated_outcome = treated.iloc[indices[i]][outcome].item()
        atc += treated_outcome - control_outcome
    atc /= numcontrolunits

    numcontrolunits = numtreatedunits + numcontrolunits
    est = (att*numtreatedunits + atc*numcontrolunits)/(numtreatedunits+numcontrolunits)
    return att, atc, est
