from dice_ml.utils import helpers # helper functions
from dice_ml.data_interfaces import public_data_interface
import numpy as np
import tensorflow as tf
from dice_ml.model_interfaces import keras_tensorflow_model
from dice_ml.dice_interfaces import dice_tensorflow2
if __name__ == "__main__":

	data = helpers.load_adult_income_dataset()
	# params = {}
	# params['dataframe'] = data
	# params['continuous_features'] = ['age', 'hours_per_week']
	# params['outcome_name'] = 'income'
	#
	# data_inter = public_data_interface.PublicData(params)
	# model_path = helpers.get_adult_income_modelpath()
	#
	# print("Loading model ")
	# model = keras_tensorflow_model.KerasTensorFlowModel(model_path=model_path)
	# print("Loading CE")
	# dice = dice_tensorflow2.DiceTensorFlow2(data_inter, model)
	#
	# query_instance = {'age':22,
    # 'workclass':'Private',
    # 'education':'HS-grad',
    # 'marital_status':'Single',
    # 'occupation':'Service',
    # 'race': 'White',
    # 'gender':'Female',
    # 'hours_per_week': 45}
	# print("Fuck ")
	# print("Hello world",dice.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite"))
	# # print(dice.compute_yloss())
	#
	# #     # Generate counterfactual examples
	# # dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite")
	# # # Visualize counterfactual explanation
	# # dice_exp.visualize_as_dataframe()
	#
	#
	#
	# # model.load_model()
	# # minx, maxx = data_inter.get_minx_maxx(normalized=False)
	#
	# # cfs = []
	# # total_CFs = 6
	#
	# # if len(cfs) == 0:
	# #     for ix in range(total_CFs):
	# #         one_init = [[]]
	# #         for jx in range(minx.shape[1]):
	# #             one_init[0].append(np.random.uniform(minx[0][jx],maxx[0][jx]))
	# #         cfs.append(tf.Variable(one_init, dtype=tf.float32))
	#
	#
	# # target_cf_class = 1
	# # yloss = 0.0
	# # yloss_type = "l2_loss"
	# # num_ouput_nodes = 10
	# # for i in range(total_CFs):
	# # 	print(cfs[i])
	# # 	print(model.get_output(cfs[i]))
	# # 	temp_loss = tf.pow((model.get_output(cfs[i]) - target_cf_class), 2)
	#
	# # 	print("Loss {}".format(temp_loss))
	# # 	temp_loss = temp_loss[:,(num_ouput_nodes-1):][0][0]
	#
	# # 	yloss += temp_loss
	#
	# # yloss = yloss/total_CFs
