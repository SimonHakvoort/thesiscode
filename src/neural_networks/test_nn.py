import numpy as np
from neural_networks.get_data import normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features, get_tf_data
from src.models.get_data import get_tensors
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf

neighbourhood_size = 5
feature_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

feature_names_dict = {name: 1 for name in feature_names}

feature_names_dict['wind_speed'] = 1

fold = 1
ignore = ['229', '285', '323']

X_1, y_1 = get_tensors(neighbourhood_size, feature_names, fold, ignore = ignore)

fold = 2

X_2, y_2 = get_tensors(neighbourhood_size, feature_names, fold, ignore = ignore)

X = np.concatenate((X_1, X_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)

test_data = get_tf_data([1,2], feature_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data, mean, std = normalize_1d_features(test_data)

test_data = test_data.shuffle(len(test_data))

test_data = test_data.batch(32)

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)





forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

loss_function = 'loss_twCRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function_mean = 12
chain_function_std = 2
chain_function_constant = 0.3

optimizer = 'adam'
learning_rate = 0.0002

dense_l2_regularization = 0.002
hidden_units_list = [100, 100, 100, 100]
add_forecast_layer = True

setup_distribution = {
    'forecast_distribution': forecast_distribution,
    'distribution_1': distribution_1,
    'distribution_2': distribution_2,
}

setup_nn_architecture = {
    'hidden_units_list': hidden_units_list,
    'dense_l2_regularization': dense_l2_regularization,
    'add_forecast_layer': add_forecast_layer,
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
    'feature_names': feature_names,
    'setup_loss': setup_loss,
    'setup_optimizer': setup_optimizer,
    'sample_size': 100,
    'setup_nn_architecture': setup_nn_architecture,
}


nn = NNForecast(**setup)

history = nn.fit(test_data, epochs=200, batch_size=32)

fold = 3
test_data = get_tf_data([fold], feature_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data = normalize_1d_features_with_mean_std(test_data, mean, std)

test_data = test_data.batch(32)

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

print(nn.CRPS(test_data, 1000))

