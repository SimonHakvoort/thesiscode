

# possible parameters: 'wind_speed', 'press', 'kinetic', 'humid', 'geopot'
# IMPORTANT: the first parameter should be 'wind_speed', this is assumed in the normalization process
import numpy as np
import tensorflow as tf
from src.models.train_emos import train_emos
from src.models.emos import EMOS
from src.models.get_data import get_tensors
from src.models.initial_params import get_gev_initial_params, get_trunc_normal_initial_params
from src.models.train_emos import train_and_test_emos
from src.training.training import load_model
from src.visualization.brier_score import make_brier_skill_plot
from src.visualization.pit import make_cpit_hist_emos
from src.visualization.reliability_diagram import make_reliability_and_sharpness
from src.visualization.scoring_tables import make_table
from src.models.probability_distributions import TruncGEV
import pickle as pkl


parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_CRPS_sample"
samples = 100

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
forecast_distribution = "distr_mixture_linear"

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


neighbourhood_size = 11
epochs = 200
test_fold = 3
folds = [1,2]
ignore = ['229', '285', '323']

#tf.debugging.enable_check_numerics()
folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/'

test_fold = 3
ignore = ['229', '285', '323']
X_test, y_test, variances_test = get_tensors(neighbourhood_size, parameter_names, test_fold, ignore)


test = load_model(folder + "mixture_linear/mixturelinear_tn_gev_twcrps_mean13.0_std1.0_constant0.009999999776482582_epochs600.pkl")

test_2 = load_model(folder + 'mixture_linear/mixturelinear_tn_gev_twcrps_mean13.0_std1.0_constant0.029999999329447746_epochs600.pkl')

test_3 = load_model(folder + 'mixture_linear/mixturelinear_tn_gev_twcrps_mean13.0_std1.5_constant0.029999999329447746_epochs600.pkl')

base_model = load_model(folder + 'trunc_normal/tn_crps_.pkl')

X_test = test.normalize_features(X_test)

test_dict = {'test_2': test_2, 'base_model': base_model}

make_reliability_and_sharpness(test_dict, X_test, y_test, variances_test, 2)


