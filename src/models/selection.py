

# possible parameters: 'wind_speed', 'press', 'kinetic', 'humid', 'geopot'
# IMPORTANT: the first parameter should be 'wind_speed', this is assumed in the normalization process
import numpy as np
from src.models.train_emos import train_emos
from src.models.emos import EMOS
from src.models.get_data import get_tensors
from src.models.initial_params import get_gev_initial_params, get_trunc_normal_initial_params
from src.models.train_emos import train_and_test_emos
from src.visualization.brier_score import brier_skill_plot


parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_CRPS_sample"
samples = 400

# possible chain functions: 'chain_function_indicator' and 'chain_function_normal_cdf'
# if chain_function_indicator is chosen, threshold is not necessary
# if chain_function_normal_cdf is chosen, threshold is necessary
chain_function = "chain_function_normal_cdf"
threshold = 8
chain_function_mean = 14
chain_function_std = 2.5


# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.01

# possible forecast distributions: 'distr_trunc_normal', 'distr_log_normal', 'distr_gev' and 'distr_mixture'/'distr_mixture_linear', which can be a mixture distribution of two previously mentioned distributions.
forecast_distribution = "distr_trunc_normal"

# necessary in case of a mixture distribution
distribution_1 = "distr_trunc_normal"
distribution_2 = "distr_gev"

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

initial_params_gev = get_gev_initial_params()
initial_params_trunc_normal = get_trunc_normal_initial_params()
parameters = {}

# for key in initial_params_gev:
#     parameters[key] = initial_params_gev[key]
# for key in initial_params_trunc_normal:
#     parameters[key] = initial_params_trunc_normal[key]

parameters['weight'] = np.array([0.5])
#setup['parameters'] = parameters

neighbourhood_size = 11
epochs = 10
test_folds = 1
folds = [2,3]

# dict = train_and_test_emos(neighbourhood_size, parameter_names, epochs, train_folds, test_folds, setup)
emos = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)
setup['forecast_distribution'] = "distr_log_normal"
emos2 = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)

X_test, y_test, variances_test = get_tensors(neighbourhood_size, parameter_names, test_folds)
X_test = (X_test - emos.feature_mean) / emos.feature_std

dict_emos = {'emos1': emos, 'emos2': emos2}
emos = dict_emos.pop('emos1')
brier_skill_plot(emos, dict_emos, X_test, y_test, variances_test, np.linspace(0, 20, 2000))

