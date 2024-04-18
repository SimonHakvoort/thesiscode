import numpy as np
from src.models.get_data import get_tensors
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf

neighbourhood_size = 5
location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']
fold = 1
ignore = ['229', '285', '323']

X_1, y_1 = get_tensors(neighbourhood_size, location_features, fold, ignore = ignore)

fold = 2

X_2, y_2 = get_tensors(neighbourhood_size, location_features, fold, ignore = ignore)

X = np.concatenate((X_1, X_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)



# set the seed for tensorflow probability
# tf.random.set_seed(42)
# np.random.seed(42)

loss_function = 'loss_twCRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function_mean = 12
chain_function_std = 2
chain_function_constant = 0.3


dense_l2_regularization = 0.0002
input_shape = X.shape[1]
hidden_layers = 3
hidden_units_list = [100, 100, 100]

nn_architecture = {
    'input_shape': input_shape,
    'hidden_layers': hidden_layers,
    'hidden_units_list': hidden_units_list,
    'dense_l2_regularization': dense_l2_regularization,
    'normalization': 'standard'
}

setup = {
    'loss_function': loss_function,
    'chain_function': chain_function,
    'chain_function_mean': chain_function_mean,
    'chain_function_std': chain_function_std,
    'chain_function_constant': chain_function_constant,
}


nn = NNForecast(nn_architecture, 'distr_trunc_normal', 100, **setup)

nn.fit(X, y, epochs=250, batch_size=32)

fold = 3
X_test, y_test = get_tensors(neighbourhood_size, location_features, fold, ignore = ignore)

print(nn.CRPS(X_test, y_test, 5000).numpy())