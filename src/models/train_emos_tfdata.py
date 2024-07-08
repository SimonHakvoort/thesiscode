import os
# Set environment variable to suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from src.neural_networks.get_data import get_tf_data, load_cv_data, load_train_test_data, make_importance_sampling_dataset, normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features
from src.models.train_emos import train_emos
from src.models.emos import LinearEMOS, BootstrapEmos
from src.models.get_data import get_tensors
from src.models.train_emos import train_and_test_emos
from src.training.training import load_model
from src.visualization.brier_score import make_brier_skill_plot
from src.visualization.pit import comp_pit_score_tf, make_cpit_hist_emos
from src.visualization.reliability_diagram import make_reliability_and_sharpness
from src.visualization.scoring_tables import make_table
from src.models.probability_distributions import TruncGEV
import pickle as pkl
import time


all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in all_features}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']


train_data, test_data, data_info = load_cv_data(1, features_names_dict)

def addweight(X, y):
    return X, y, tf.constant(1, dtype=tf.float32)

train_data = train_data.map(addweight)

train_data = train_data.shuffle(train_data.cardinality())

train_data = train_data.batch(256)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.batch(test_data.cardinality())

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)


# possible loss functions: 'loss_CRPS_sample', 'loss_log_likelihood', 'loss_Brier_score', 'loss_twCRPS_sample'
loss = "loss_twCRPS_sample"
#loss = "loss_cPIT"
samples = 250

# possible chain functions: 'chain_function_indicator' and 'chain_function_normal_cdf'
# if chain_function_indicator is chosen, threshold is not necessary
# if chain_function_normal_cdf is chosen, threshold is necessary
chain_function = "chain_function_normal_cdf_plus_constant"
threshold = 8
chain_function_mean = 8.830960
chain_function_std = 1.068426
chain_function_constant = 0.015801

		
				

# possible optimizers: 'SGD', 'Adam'
optimizer = "Adam"
learning_rate = 0.01

# possible forecast distributions: 'distr_trunc_normal', 'distr_log_normal', 'distr_gev' and 'distr_mixture'/'distr_mixture_linear', which can be a mixture distribution of two previously mentioned distributions.
forecast_distribution = "distr_trunc_normal"

# necessary in case of a mixture distribution
distribution_1 = "distr_trunc_normal"
distribution_2 = "distr_log_normal"

random_init = False
printing = True
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
         'all_features': all_features,
         'location_features': location_features,
         'scale_features': scale_features,
         'random_init': random_init,
         'subset_size': subset_size,
        'printing': printing,
         }

if forecast_distribution == 'distr_mixture_linear' or forecast_distribution == 'distr_mixture':
    setup['forecast_distribution'] = distribution_1

    emos1 = LinearEMOS(setup)

    setup['forecast_distribution'] = distribution_2

    emos2 = LinearEMOS(setup)

    emos1.fit(train_data, 75, False)
    emos2.fit(train_data, 75, False)

    setup['forecast_distribution'] = forecast_distribution
    print(setup['forecast_distribution'])

    setup['parameters'] = {**emos1.get_parameters(), **emos2.get_parameters()}

#save the model:
# filepath = '/net/pc200239/nobackup/users/hakvoort/models/bootstrap_emos/tn_ln_M13_STD2_C07'
filepath = '/net/pc200239/nobackup/users/hakvoort/models/emos_tf/testing'

epochs = 20


batch_size = None

emos = LinearEMOS(setup)

my_dict = emos.fit(train_data, epochs)


print(emos.CRPS(test_data, 10000))

print(emos.forecast_distribution.get_gev_shape())

mydict = emos.to_dict()

with open(filepath, 'wb') as f:
    pkl.dump(mydict, f)

# print(emos.CRPS(test_data, 10000))
# print(emos.twCRPS(test_data, [12], 10000))

# bootstrap = BootstrapEmos.load(filepath)

#bootstrap = BootstrapEmos(setup, filepath, epochs, batch_size, cv, features_names_dict)

# bootstrap.num_models = 1000

# bootstrap.save_bootstrap_info()

# bootstrap.train_models(1000 - bootstrap.num_models)

# twCRPS = bootstrap.CRPS(test_data, 1000)

# print(twCRPS)

# print(np.mean(twCRPS))


