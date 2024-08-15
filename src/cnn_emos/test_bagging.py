import time
import keras
import numpy as np
from src.loading_data.get_data import get_fold_is, load_cv_data, load_train_test_data, make_importance_sampling_dataset, normalize_1d_features, normalize_1d_features_with_mean_std, save_cv_data, stack_1d_features, get_tf_data
from src.cnn_emos.nn_model import NNModel
from src.cnn_emos.nn_forecast import CNNEMOS,  CNNBaggingEMOS
import tensorflow as tf
import pickle
import os
import random


features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

bounds = {7.5: 1, 9: 3, 12: 4, 15: 9, 100: 15}


# seed = 100

# tf.random.set_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

batch_size = 64

train_data0, test_data0, data_info = load_cv_data(0, features_names_dict)


 
forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

loss_function = 'loss_twCRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
# chain_function_mean = 8.84
# chain_function_std = 1.07
# chain_function_constant = 0.015

chain_function_mean = 5.419507
chain_function_std = 7.822199
chain_function_constant = 0.919453

optimizer = 'adam'
learning_rate = 0.000105

dense_l1_regularization = 0.000
dense_l2_regularization = 0.031658
hidden_units_list = [170, 170]
conv_7x7_units = 4
conv_5x5_units = 4
conv_3x3_units = 4

metrics = ['twCRPS_12']# ['twCRPS_10', 'twCRPS_12', 'twCRPS_15']
metrics = None
saving = True

epochs = 130

setup_distribution = {
    'forecast_distribution': forecast_distribution,
    'distribution_1': distribution_1,
    'distribution_2': distribution_2,
}

setup_nn_architecture = {
    'hidden_units_list': hidden_units_list,
    'dense_l1_regularization': dense_l1_regularization,
    'dense_l2_regularization': dense_l2_regularization,

    'conv_7x7_units': conv_7x7_units,
    'conv_5x5_units': conv_5x5_units,
    'conv_3x3_units': conv_3x3_units,
}

setup_loss = {
    'loss_function': loss_function,
    'chain_function': chain_function,
    'chain_function_mean': chain_function_mean,
    'chain_function_std': chain_function_std,
    'chain_function_constant': chain_function_constant,
}

setup_optimizer = {
    'optimizer': optimizer,
    'learning_rate': learning_rate,
}

setup = {
    'setup_distribution': setup_distribution,
    'features_names': features_names,
    'setup_loss': setup_loss,
    'setup_optimizer': setup_optimizer,
    'sample_size': 1000,
    'setup_nn_architecture': setup_nn_architecture,

    'features_1d_mean': data_info['features_1d_mean'],
    'features_1d_std': data_info['features_1d_std'],
    'metrics': metrics,
}

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/bagging_cnn/bagging_model_87_tn'
# bagging_base
# bagging_model_87_mixture
# bagging_model_87_tn
# bagging_extreme_mixture

size = 10

bagging = CNNBaggingEMOS(setup, size, filepath)

bagging.train_and_save_models(train_data0, epochs = 47, batch_size = batch_size)



# bagging = CNNBaggingEMOS.my_load(filepath)

# bagging.load_models(train_data0)

# bs = bagging.twCRPS(test_data0, [0,1,2,3,4,5])

# x = 3