from src.neural_networks.get_data import get_tf_data, stack_1d_features, normalize_1d_features_with_mean_std, load_cv_data
from src.neural_networks.nn_forecast import NNForecast
from src.visualization.twcrpss_plot import make_twcrpss_plot_tf
from src.visualization.brier_score import make_brier_skill_plot_tf
from src.visualization.pit import comp_multiple_pit_scores, make_cpit_diagram_tf
from src.training.training import load_model

import tensorflow as tf
import numpy as np



features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data, test_data, data_info = load_cv_data(3, features_names_dict)

train_data = train_data.shuffle(len(train_data))

train_data = train_data.batch(32)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data_original = test_data



filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/'

crps_tn_e10 = NNForecast.my_load(filepath + 'CRPS_trunc_normal_epochs_10', train_data)

crps_tn_e10_2 = NNForecast.my_load(filepath + 'CRPS_trunc_normal_epochs_10_v2', train_data)

crps_tn_e10_3 = NNForecast.my_load(filepath + 'CRPS_trunc_normal_epochs_10_v3', train_data)

crps_tn_e10_4 = NNForecast.my_load(filepath + 'CRPS_trunc_normal_epochs_10_v4', train_data)


nn_crps_e10_models = {'CRPS_trunc_normal_epochs_10': crps_tn_e10,
                      'CRPS_trunc_normal_epochs_10_v2': crps_tn_e10_2,
                      'CRPS_trunc_normal_epochs_10_v3': crps_tn_e10_3,
                      'CRPS_trunc_normal_epochs_10_v4': crps_tn_e10_4}



filepath = '/net/pc200239/nobackup/users/hakvoort/models/emos/batching/'

emos_base = load_model(filepath + 'crps_batch_none_epochs_600')


t = 15
make_cpit_diagram_tf(nn_crps_e10_models, test_data_original, t=t, base_model=emos_base)

pits_scores = comp_multiple_pit_scores(nn_crps_e10_models, test_data_original, t=t, base_model=emos_base)

    


