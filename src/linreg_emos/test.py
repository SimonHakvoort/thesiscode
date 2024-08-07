from src.cnn_emos.get_data import get_tf_data, stack_1d_features, normalize_1d_features_with_mean_std, load_cv_data
from src.cnn_emos.nn_forecast import CNNEMOS
from src.visualization.twcrpss_plot import make_twcrpss_plot_tf
from src.visualization.brier_score import make_bootstrap_sample, make_brier_skill_plot_tf
from src.visualization.pit import make_cpit_diagram_tf, comp_multiple_pit_scores
from src.visualization.reliability_diagram import make_reliability_and_sharpness_tf
from src.visualization.plot_forecasts import plot_forecast_pdf_tf
from src.climatology.climatology import Climatology
from src.linreg_emos.emos import BootstrapEmos, LinearEMOS


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import optuna

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data_original, test_data_original, data_info = load_cv_data(3, features_names_dict)

train_data = train_data_original.shuffle(len(train_data_original))

train_data = train_data.batch(32)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data_original.batch(len(test_data_original))

test_data = test_data.repeat()

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

filepath = '/net/pc200239/nobackup/users/hakvoort/models/emos_tf/base_emos_fold_3'


with open(filepath, 'rb') as f:
    emos_base = LinearEMOS(pickle.load(f))

filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/'

model_115_3 = CNNEMOS.my_load(filepath + 'CRPS_mixture_epochs_100run_115_fold_3', train_data)

values = np.linspace(0,20, 20)
base_brier = emos_base.Brier_Score(test_data, values)

model115_brier = model_115_3.Brier_Score(test_data, values)

filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/'
us1 = CNNEMOS.my_load(filepath + 'twCRPS_mean_9_std_0.25_constant_0.01_trunc_normal_epochs_200_us_fixed_seed_1', train_data)

us_brier = us1.Brier_Score(test_data, values)

print(1 - model115_brier / base_brier)

