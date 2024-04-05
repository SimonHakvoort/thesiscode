import pickle as pkl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.models.train_emos import train_emos, train_and_test_emos
from src.training.training import load_model, train_and_save
from src.visualization.brier_score import make_brier_skill_plot, brier_plot
from src.visualization.pit import make_cpit_diagram_emos
from src.visualization.plot_forecasts import plot_forecast_pdf
from src.visualization.reliability_diagram import make_reliability_diagram
from src.models.get_data import get_tensors, get_normalized_tensor
from src.models.emos import EMOS
from src.models.emos import EMOS

forecast_distribution = 'distr_mixture_linear'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_gev'

loss = 'loss_twCRPS_sample' # options: loss_CRPS_sample, loss_twCRPS_sample, loss_log_likelihood

chain_function = 'chain_function_normal_cdf' # options: chain_function_normal_cdf, chain_function_indicator, chain_function_normal_cdf_plus_constant
chain_function_mean = 13
chain_function_std = 2
chain_function_threshold = 15 # 12 / 15
chain_function_constant = 0.02

optimizer = 'Adam'
learning_rate = 0.03
folds = [1,2]
neighbourhood_size = 11
ignore = ['229', '285', '323']
epochs = 600

samples = 100
printing = True
pretrained = True
random_init = False

all_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot', 'spatial_variance']

location_features = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

scale_features = ['spatial_variance']

model = train_and_save(
    forecast_distribution,
    loss,
    optimizer,
    learning_rate,
    folds,
    all_features,
    location_features,
    scale_features,
    neighbourhood_size,
    ignore,
    epochs,

    chain_function = chain_function,
    chain_function_mean = chain_function_mean,
    chain_function_std = chain_function_std,
    chain_function_constant = chain_function_constant,
    chain_function_threshold = chain_function_threshold,
    samples = samples,
    printing = printing,
    distribution_1 = distribution_1,
    distribution_2 = distribution_2,
    pretrained = pretrained,
    random_init = random_init
)