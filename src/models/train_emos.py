import os
import pickle as pkl
import tensorflow as tf

from src.models.emos import EMOS
from src.models.get_data import get_normalized_tensor, get_tensors



def train_emos(neighbourhood_size, parameter_names, epochs, folds, setup):
    
    data = get_normalized_tensor(neighbourhood_size, parameter_names, folds)

    X = data["X"]
    y = data["y"]
    variances = data["variances"]

    setup["num_features"] = len(parameter_names)
    setup["feature_mean"] = data["mean"]
    setup["feature_std"] = data["std"]
    setup["features"] = parameter_names
    setup["neighbourhood_size"] = neighbourhood_size

    emos = EMOS(setup)
    emos.fit(X, y, variances, epochs)

    return emos

def train_and_save_emos(neighbourhood_size, parameter_names, epochs, folds, setup):
    emos = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)
    mydict = emos.to_dict()
    with open('/net/pc200239/nobackup/users/hakvoort/models/emos_{0}_{1}.pkl'.format(setup['loss'], setup['forecast_distribution']), 'wb') as f:
        pkl.dump(mydict, f)
    
    print("Model saved")
    

# possible parameters: 'wind_speed', 'press', 'kinetic', 'humid', 'geopot'
# IMPORTANT: the first parameter should be 'wind_speed', this is assumed in the normalization process
parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_twCRPS_sample"
samples = 50

# possible chain functions: 'chain_function_indicator' and 'chain_function_normal_cdf'
# if chain_function_indicator is chosen, threshold is not necessary
# if chain_function_normal_cdf is chosen, threshold is necessary
chain_function = "chain_function_normal_cdf"
threshold = 8
chain_function_mean = 9
chain_function_std = 2


# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.01

# possible forecast distributions: 'distr_trunc_normal', 'distr_log_normal' and 'distr_mixture', which can be a mixture distribution of two previously mentioned distributions.
forecast_distribution = "distr_mixture"

# necessary in case of a mixture distribution
distribution_1 = "distr_trunc_normal"
distribution_2 = "distr_log_normal"

setup = {'loss': loss,
         'samples': samples, 
         'optimizer': optimizer, 
         'learning_rate': learning_rate, 
         'forecast_distribution': forecast_distribution,
         'chain_function': chain_function,
         'threshold': threshold,
         'distribution_1': distribution_1,
         'distribution_2': distribution_2,
         'chain_function_mean': chain_function_mean,
         'chain_function_std': chain_function_std
         }

neighbourhood_size = 11
epochs = 320
folds = [1,2,3]

#tf.debugging.enable_check_numerics()

emos = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)

mydict = emos.to_dict()

emos2 = train_emos(neighbourhood_size, parameter_names, epochs, folds, mydict)
print(emos2)
