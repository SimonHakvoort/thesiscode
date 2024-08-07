import pickle as pkl
import numpy as np
from itertools import chain

from src.linreg_emos.train_emos import train_emos, train_and_test_emos
from src.visualization.pit import make_cpit_diagram_emos, make_cpit_hist_emos 
from src.visualization.brier_score import make_brier_skill_plot, brier_plot
from src.linreg_emos.get_data import get_tensors, get_normalized_tensor
from src.linreg_emos.emos import LinearEMOS
from src.visualization.scoring_tables import make_table
from src.visualization.twcrpss_plot import make_twcrpss_plot
from src.linreg_emos.emos import LinearEMOS
from src.visualization.reliability_diagram import make_reliability_diagram, make_reliability_diagram_sklearn
from src.training.training import load_model, train_model
from src.visualization.plot_forecasts import plot_forecast_cdf, plot_forecast_pdf, plot_weight_mixture

forecast_distribution = 'distr_mixture_linear'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_gev'

loss = 'loss_twCRPS_sample' # options: loss_CRPS_sample, loss_twCRPS_sample, loss_log_likelihood

chain_function = 'chain_function_normal_cdf_plus_constant' # options: chain_function_normal_cdf, chain_function_indicator, chain_function_normal_cdf_plus_constant
chain_function_mean = 13
chain_function_std = 1
chain_function_threshold = 15 # 12 / 15
chain_function_constant = 0.01

optimizer = 'Adam'
learning_rate = 0.01
folds = [1,2]
parameter_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']
neighbourhood_size = 11
ignore = ['229', '285', '323']
epochs = 40

samples = 200
printing = True
pretrained = True

model = train_model(
    forecast_distribution,
    loss,
    optimizer,
    learning_rate,
    folds,
    parameter_names,
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
    pretrained = pretrained
)