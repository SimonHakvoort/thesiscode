from src.loading_data.get_data import load_cv_data
from src.cnn_emos.nn_forecast import CNNBaggingEMOS

# Load the training data.
features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

bounds = {7.5: 1, 9: 3, 12: 4, 15: 9, 100: 15}

batch_size = 64

train_data, test_data, data_info = load_cv_data(2, features_names_dict)

# The train data should not be shuffled and batched, since this will be done during the optimization process.


# Select the parametric distribution. The variables distribution_1 and distribution_2 only get used in case a mixture distibution is selected. 
forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

# Choice of loss function. In case the twCRPS is used, a weight function including its parameters should be selected.
loss_function = 'loss_CRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
# chain_function = 'chain_function_indicator'
chain_function_threshold = 12

### Sharp sigmoid parameters
# chain_function_mean = 8.84
# chain_function_std = 1.07
# chain_function_constant = 0.015

### Best CNN parameters
chain_function_mean = 5.419507
chain_function_std = 7.822199
chain_function_constant = 0.919453

### Sigmoid parameters
# chain_function_mean = 7.050563812255859
# chain_function_std = 2.405172109603882
# chain_function_constant = 0.06170300021767616	


optimizer = 'adam'
learning_rate = 0.000105

dense_l1_regularization = 0.000
dense_l2_regularization = 0.031658
hidden_units_list = [170, 170]
conv_7x7_units = 4
conv_5x5_units = 4
conv_3x3_units = 4

# Metrics that should be displayed during the optimization process. Options are twCRPS12
metrics = ['twCRPS_12']
metrics = None

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
    'chain_function_threshold': chain_function_threshold
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

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/bagging_same_data/tn_crps_fold2'

size = 10

# Make a CNNBaggingEMOS instance and train the models.
bagging = CNNBaggingEMOS(setup, size, filepath,  bootstrap_training_data=False)

bagging.train_and_save_models(train_data, epochs = 71, batch_size = batch_size)
