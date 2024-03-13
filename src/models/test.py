import pickle as pkl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.train_emos import train_emos, train_and_test_emos
from src.training.training import load_model
from src.visualization.brier_score import brier_skill_plot, brier_plot
from src.visualization.pit import make_cpit_diagram_emos
from src.visualization.plot_forecasts import plot_forecast_pdf
from src.visualization.reliability_diagram import make_reliability_diagram
from src.models.get_data import get_tensors, get_normalized_tensor
from src.models.emos import EMOS
from src.models.emos import EMOS

# neighbourhood_size = 11
# parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']
# ignore = ['229', '285', '323']
# train_folds = [1, 2]
# train_data = get_normalized_tensor(neighbourhood_size, parameter_names, train_folds, ignore)

# X_train = train_data['X']
# y_train = train_data['y']
# variances_train = train_data['variances']
# mean_train = train_data['mean']
# std_train = train_data['std']

# test_fold = 3

# X_test, y_test, variances_test = get_tensors(neighbourhood_size, parameter_names, test_fold, ignore)
# X_test = (X_test - mean_train) / std_train

# print(X_test.shape)

# # load the dictionary with the data from /net/pc200239/nobackup/users/hakvoort/models
# with open('/net/pc200239/nobackup/users/hakvoort/models/emos_crps.pkl', 'rb') as f:
#     models_crps = pkl.load(f)

# emos_crps = {}

# #emos_crps["crps_log_normal"] = EMOS(models_crps["crps_log_normal"])
# emos_crps["crps_tn"] = EMOS(models_crps["crps_tn"])

# value = 5
# make_reliability_diagram(emos_crps, X_test, y_test, variances_test, value, n_subset = 11)
tfd = tfp.distributions

mix = tf.Variable(0.5, dtype=tf.float32)
cat = tfd.Categorical(probs=[mix, 1.-mix])
bimix_gauss = tfd.Mixture(
    cat=cat,
    components=[
      tfd.Normal(loc=-10., scale=1),
      tfd.Normal(loc=+10., scale=1),
  ])

samples = bimix_gauss.sample(1000)
for i in range(1000):
    print(samples[i].numpy())



# folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/trunc_normal/'

# mean12_std4 = load_model(folder + 'tn_twcrps_mean12.0_std4.0.pkl')
# mean12_std1 = load_model(folder + 'tn_twcrps_mean12.0_std1.0.pkl')
# mean15_std1 = load_model(folder + 'tn_twcrps_mean15.0_std1.0.pkl')
# mean16_std5 = load_model(folder + 'tn_twcrps_mean16.0_std5.0.pkl')

# model_dict = {'mean12_std4': mean12_std4, 'mean12_std1': mean12_std1, 'mean15_std1': mean15_std1, 'mean16_std5': mean16_std5}

# tn_models = {'mean12_std4': mean12_std4, 'mean12_std1': mean12_std1, 'mean15_std1': mean15_std1, 'mean16_std5': mean16_std5}

# folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/'

# base_model = load_model(folder + 'trunc_normal/tn_crps_.pkl')

# test_fold = 3
# ignore = ['229', '285', '323']
# X_test, y_test, variances_test = get_tensors(mean12_std4.neighbourhood_size, mean12_std4.feature_names, test_fold, ignore)
# X_test = (X_test - mean12_std4.feature_mean) / mean12_std4.feature_std

# t=15
# make_cpit_diagram_emos(model_dict, X_test, y_test, variances_test, t=t, base_model=base_model)