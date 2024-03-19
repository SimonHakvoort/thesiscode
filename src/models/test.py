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

folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/'

mixture = load_model(folder + 'mixture_linear/mixturelinear_tn_gev_twcrps_mean13.0_std2.0_epochs600.pkl')

test_fold = 3
ignore = ['229', '285', '323']
X_test, y_test, variances_test = get_tensors(mixture.neighbourhood_size, mixture.feature_names, test_fold, ignore)
X_test = (X_test - mixture.feature_mean) / mixture.feature_std

distribution = mixture.forecast_distribution.get_distribution(X_test, variances_test)
weight = distribution.weight.numpy()

min_index = np.argmin(weight)

cdf = distribution.cdf(y_test)