import time
import numpy as np
from src.loading_data.get_data import load_cv_data
from src.cnn_emos.nn_forecast import CNNEMOS
import tensorflow as tf
import pickle

# Load the data for the folds
features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

bounds = {7.5: 1, 9: 3, 12: 4, 15: 9, 100: 15}


batch_size = 64

train_data0, test_data0, data_info = load_cv_data(0, features_names_dict)

dataset_size = train_data0.cardinality()

train_data0 = train_data0.shuffle(train_data0.cardinality())

train_data0 = train_data0.batch(batch_size)

train_data0 = train_data0.prefetch(tf.data.experimental.AUTOTUNE)

test_data0 = test_data0.batch(len(test_data0))

test_data0 = test_data0.prefetch(tf.data.experimental.AUTOTUNE)


 # Select the parametric distribution. The variables distribution_1 and distribution_2 only get used in case a mixture distibution is selected. 
forecast_distribution = 'distr_mixture'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

# Choice of loss function. In case the twCRPS is used, a weight function including its parameters should be selected.
loss_function = 'loss_CRPS_sample'
# chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function = 'chain_function_indicator'
chain_function_threshold = 12
chain_function_mean = 8.84
chain_function_std = 1.07
chain_function_constant = 0.015

# chain_function_mean = 5.419507
# chain_function_std = 7.822199
# chain_function_constant = 0.919453

optimizer = 'adam'
learning_rate = 0.000105

dense_l1_regularization = 0.000
dense_l2_regularization = 0.031658
hidden_units_list = [170, 170]
conv_7x7_units = 4
conv_5x5_units = 4
conv_3x3_units = 4

# Display metrics during the training process. Options are twCRPS_10, twCRPS_12, twCRPS_15 and CRPS
metrics = ['twCRPS_12']

saving = False

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
    'chain_function_threshold': chain_function_threshold,
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

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/cnnemos/crps_tn_variance_5'

if saving:
    with open(filepath + '/attributes', 'wb') as f:
        pickle.dump(setup, f)

nn = CNNEMOS(**setup)

epochs = 6

history = nn.fit(train_data0, epochs=epochs)

x = nn.CRPS(test_data0)
print(x)
if saving:
    nn.save_weights(filepath)
    print("Model saved")
