

# possible parameters: 'wind_speed', 'press', 'kinetic', 'humid', 'geopot'
# IMPORTANT: the first parameter should be 'wind_speed', this is assumed in the normalization process
import numpy as np
import tensorflow as tf
from src.models.train_emos import train_emos
from src.models.emos import EMOS
from src.models.get_data import get_tensors
from src.models.initial_params import get_gev_initial_params, get_trunc_normal_initial_params
from src.models.train_emos import train_and_test_emos
from src.visualization.brier_score import brier_skill_plot
from src.visualization.pit import make_cpit_hist_emos
from src.visualization.scoring_tables import make_table
from src.models.probability_distributions import TruncGEV
import pickle as pkl


parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_twCRPS_sample"
samples = 200

# possible chain functions: 'chain_function_indicator' and 'chain_function_normal_cdf'
# if chain_function_indicator is chosen, threshold is not necessary
# if chain_function_normal_cdf is chosen, threshold is necessary
chain_function = "chain_function_normal_cdf"
threshold = 8
chain_function_mean = 14
chain_function_std = 1


# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.01

# possible forecast distributions: 'distr_trunc_normal', 'distr_log_normal', 'distr_gev' and 'distr_mixture'/'distr_mixture_linear', which can be a mixture distribution of two previously mentioned distributions.
forecast_distribution = "distr_trunc_gev"

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



neighbourhood_size = 7
epochs = 400
test_fold = 3
folds = [1,2]
ignore = ['229', '285', '323']

tf.debugging.enable_check_numerics()

emos = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)
print(emos)
# print(emos)





