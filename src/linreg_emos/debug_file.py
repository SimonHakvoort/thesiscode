from src.loading_data.get_data import get_tf_data, stack_1d_features, normalize_1d_features_with_mean_std, load_cv_data
from src.cnn_emos.nn_forecast import CNNEMOS
from src.visualization.twcrpss_plot import make_twcrpss_plot_tf
from src.visualization.brier_score import make_bootstrap_brier_efficient, make_bootstrap_sample, make_brier_skill_plot_tf
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

all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in all_features}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data0, test_data0_original, ignore = load_cv_data(0, features_names_dict)

data_load = train_data0.batch(32)

train_data0 = train_data0.batch(train_data0.cardinality())
test_data0 = test_data0_original.batch(test_data0_original.cardinality())

folder = '/net/pc200239/nobackup/users/hakvoort/models/climatology/climatology_cv_0'

climatology = Climatology.load(folder)

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/linregemos/emos_base'

with open(filepath, 'rb') as f:
    emos_base = LinearEMOS(pickle.load(f))

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/linregemos/extreme'

with open(filepath, 'rb') as f:
    extreme = LinearEMOS(pickle.load(f))

filepath = '/net/pc200239/nobackup/users/hakvoort/models/final_models/linregemos/intermediate'

with open(filepath, 'rb') as f:
    intermdiate = LinearEMOS(pickle.load(f))

linear_weight_models = {'Extreme': extreme, 'Intermediate': intermdiate}

all_linear_models = {'Extreme': extreme, 'Intermediate': intermdiate, 'Vanilla': emos_base}

ylim = [-0.1,0.1]
values = np.linspace(0,20,20)
make_bootstrap_brier_efficient(emos_base, linear_weight_models, test_data0, values, ylim=ylim, bootstrap_size=1000, name_base_model='Vanilla')

