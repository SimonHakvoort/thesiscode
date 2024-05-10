import numpy as np
import tensorflow as tf
from neural_networks.get_data import get_tf_data, normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features
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


all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in all_features}




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
         'all_features': all_features,
         'location_features': location_features,
         'scale_features': scale_features,
         'random_init': random_init,
         'subset_size': subset_size,
        'printing': printing,
         }


neighbourhood_size = 11
epochs = 100
test_fold = 3

ignore = ['229', '285', '323']

train_data = get_tf_data([1,2], features_names_dict, ignore=ignore)

train_data = train_data.map(lambda x, y: stack_1d_features(x, y))

train_data, mean, std = normalize_1d_features(train_data)

train_data = train_data.shuffle(len(train_data))

train_data = train_data.batch(64)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

setup['feature_mean'] = mean
setup['feature_std'] = std

emos = EMOS(setup)
#start timing:
start = time.time()

print("Starting training")

loss = emos.fit_tfdataset(train_data, epochs, printing = False)

#end timing:
end = time.time()


print("Time taken to train model: ", end - start)

fold = 3
test_data = get_tf_data([fold], features_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data = normalize_1d_features_with_mean_std(test_data, mean, std)

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

print(emos.CRPS_tfdataset(test_data, 1000))

print(emos.Brier_Score_tfdataset(test_data, 10))
print(emos.Brier_Score_tfdataset(test_data, 15))

print(emos.twCRPS_tfdataset(test_data, 12, 1000))