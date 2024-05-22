
import time
import keras
import numpy as np
from src.neural_networks.get_data import load_cv_data, load_train_test_data, normalize_1d_features, normalize_1d_features_with_mean_std, save_cv_data, stack_1d_features, get_tf_data
from src.neural_networks.nn_model import NNModel
from src.models.get_data import get_tensors
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf
import pickle
import os

from src.visualization.pit import comp_pit_score_tf

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']


train_data, test_data, data_info = load_cv_data(3, features_names_dict)

train_data = train_data.shuffle(len(train_data))

train_data = train_data.batch(32)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

loss_function = 'loss_CRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function_mean = 10
chain_function_std = 1
chain_function_constant = 0.1

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

metrics = ['twCRPS_12']# ['twCRPS_10', 'twCRPS_12', 'twCRPS_15']

saving = True

epochs = 200

filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/'

if loss_function == 'loss_twCRPS_sample':
    name = 'twCRPS'
    name += '_mean_' + str(chain_function_mean)
    name += '_std_' + str(chain_function_std)
    name += '_constant_' + str(chain_function_constant)
elif loss_function == 'loss_CRPS_sample':
    name = 'CRPS'

filepath += name + '_'

if forecast_distribution == 'distr_mixture':
    filepath += 'mixture_'
elif forecast_distribution == 'distr_trunc_normal':
    filepath += 'trunc_normal_'
elif forecast_distribution == 'distr_log_normal':
    filepath += 'log_normal_'

filepath += 'epochs_' + str(epochs) 



#filepath += '_v2'


# make a folder
if saving:
    os.makedirs(filepath, exist_ok=True)

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

    'features_1d_mean': data_info['features_1d_mean'],
    'features_1d_std': data_info['features_1d_std'],

    'metrics': metrics,
}

if saving:
    with open(filepath + '/attributes', 'wb') as f:
        pickle.dump(setup, f)

nn = NNForecast(**setup)

#start the time
time_start = time.time()

history = nn.fit(train_data, epochs=epochs, validation_data=test_data)

#end the time
time_end = time.time()

print("Time: ", time_end - time_start)

print(nn.CRPS(test_data, 10000).numpy())

print(nn.twCRPS(test_data, [12], 10000)[0].numpy())

if saving:
    nn.save_weights(filepath)
    print("Model saved")

# save the history
if saving:
    with open(filepath + '/history.pickle', 'wb') as f:
        pickle.dump(history.history, f)
        print("History saved")



