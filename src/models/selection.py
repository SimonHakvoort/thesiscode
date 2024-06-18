import numpy as np
import tensorflow as tf
from src.models.train_emos import train_emos
from src.models.emos import EMOS
from src.models.get_data import get_tensors
from src.models.train_emos import train_and_test_emos
from src.training.training import load_model
from src.visualization.brier_score import make_brier_skill_plot
from src.visualization.pit import make_cpit_hist_emos
from src.visualization.reliability_diagram import make_reliability_and_sharpness
from src.visualization.scoring_tables import make_table
from src.models.probability_distributions import TruncGEV
import pickle as pkl
import time


all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot', 'spatial_variance']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']




# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_CRPS_sample"
samples = 100

# possible chain functions: 'chain_function_indicator' and 'chain_function_normal_cdf'
# if chain_function_indicator is chosen, threshold is not necessary
# if chain_function_normal_cdf is chosen, threshold is necessary
chain_function = "chain_function_normal_cdf_plus_constant"
threshold = 8
chain_function_mean = 13
chain_function_std = 2
chain_function_constant = 0.07


# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.05

# possible forecast distributions: 'distr_trunc_normal', 'distr_log_normal', 'distr_gev' and 'distr_mixture'/'distr_mixture_linear', which can be a mixture distribution of two previously mentioned distributions.
forecast_distribution = "distr_trunc_normal"

# necessary in case of a mixture distribution
distribution_1 = "distr_trunc_normal"
distribution_2 = "distr_log_normal"

random_init = False
printing = False
subset_size = None

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
         'chain_function_std': chain_function_std,
         'chain_function_constant': chain_function_constant,
         'location_features': location_features,
         'scale_features': scale_features,
         'random_init': random_init,
         'subset_size': subset_size,
        'printing': printing,
         }


neighbourhood_size = 11
epochs = 20
test_fold = 3
folds = [1,2]
ignore = ['229', '285', '323']

# tf.debugging.enable_check_numerics()
folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/test'

# time the length that train_emos takes
start = time.time()

model = train_emos(neighbourhood_size, all_features, epochs, folds, setup, ignore=ignore)

end = time.time()
print("Time taken to train model: ", end - start)

test_fold = 3
ignore = ['229', '285', '323']
X_test, y_test = get_tensors(model.neighbourhood_size, model.all_features, test_fold, ignore)
X_test = (X_test - model.feature_mean) / model.feature_std

print(model._crps_computation(X_test, y_test, 20000))

mydict = model.to_dict()

#pickle mydict to folder

with open(folder, 'wb') as f:
    pkl.dump(mydict, f)

print("Model saved")