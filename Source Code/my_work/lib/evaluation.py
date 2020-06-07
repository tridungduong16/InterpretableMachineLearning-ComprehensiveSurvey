import numpy as np
import math
def PEHE(y1, y0, Y1, Y0):
    causal_effect = y1 - y0
    true_effect = Y1 - Y0
    return sqrt((1/n)*np.sum((causal_effect - true_effect)**2))

def MAE(true_effect, estimation_effect):
    return abs(true_effect - estimation_effect)
