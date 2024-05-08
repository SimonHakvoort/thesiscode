
import time
import keras
import numpy as np
from src.neural_networks.get_data import normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features, get_tf_data
from src.neural_networks.nn_model import NNModel
from src.models.get_data import get_tensors
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf
import pickle


neighbourhood_size = 5
features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

# nnmodel = NNModel()

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data = get_tf_data([1,2], features_names_dict, ignore=ignore)

train_data = train_data.map(lambda x, y: stack_1d_features(x, y))

train_data, mean, std = normalize_1d_features(train_data)

train_data = train_data.shuffle(len(train_data))

train_data = train_data.batch(32)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)






forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

loss_function = 'loss_twCRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function_mean = 12
chain_function_std = 2
chain_function_constant = 0.2

optimizer = 'adam'
learning_rate = 0.0002


dense_l1_regularization = 0.000
dense_l2_regularization = 0.0003
hidden_units_list = [100, 100, 100]
add_nwp_forecast = True
conv_7x7_units = 5
conv_5x5_units = 5
conv_3x3_units = 5
add_wind_conv = True

filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/test2'


setup_distribution = {
    'forecast_distribution': forecast_distribution,
    'distribution_1': distribution_1,
    'distribution_2': distribution_2,
}

setup_nn_architecture = {
    'hidden_units_list': hidden_units_list,
    'dense_l1_regularization': dense_l1_regularization,
    'dense_l2_regularization': dense_l2_regularization,
    'add_nwp_forecast': add_nwp_forecast,

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
    'sample_size': 100,
    'setup_nn_architecture': setup_nn_architecture,

    'add_wind_conv': add_wind_conv,

    'features_1d_mean': mean,
    'features_1d_std': std,
}

with open(filepath + '/attributes', 'wb') as f:
    pickle.dump(setup, f)

nn = NNForecast(**setup)

#start the time
time_start = time.time()

history = nn.fit(train_data, epochs=10)

#end the time
time_end = time.time()

print("Time: ", time_end - time_start)

fold = 3
test_data = get_tf_data([fold], features_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data = normalize_1d_features_with_mean_std(test_data, mean, std)

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

print(nn.CRPS(test_data, 1000))

nn.save_weights(filepath)

# print(nn.model.summary())




