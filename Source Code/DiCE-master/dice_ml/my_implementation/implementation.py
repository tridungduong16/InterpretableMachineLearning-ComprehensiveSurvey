#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 06:44:05 2020

@author: dtd
"""

import sys
sys.path.append('/home/dtd/Documents/interpretable_machine_learning/Source Code/DiCE-master')

from dice_ml.utils import helpers # helper functions
from dice_ml.data_interfaces import public_data_interface
import numpy as np
import tensorflow as tf
from dice_ml.model_interfaces import keras_tensorflow_model
from dice_ml.dice_interfaces import dice_tensorflow2
import sys 
if __name__ == "__main__":
    data = helpers.load_adult_income_dataset()
    params = {}
    params['dataframe'] = data
    params['continuous_features'] = ['age', 'hours_per_week']
    params['outcome_name'] = 'income'
    
    data_inter = public_data_interface.PublicData(params)
    model_path = helpers.get_adult_income_modelpath()
    print("Loading model ")
    model = keras_tensorflow_model.KerasTensorFlowModel(model_path=model_path)
    print("Loading CE")
    dice = dice_tensorflow2.DiceTensorFlow2(data_inter, model)
	#
    query_instance = {'age':22,
                      'workclass':'Private',
                      'education':'HS-grad',
                      'marital_status':'Single',
                      'occupation':'Service',
                      'race': 'White',
                      'gender':'Female',
                      'hours_per_week': 45}
    counterfactual_samples  = dice.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
    
    #print(dice.compute_yloss())