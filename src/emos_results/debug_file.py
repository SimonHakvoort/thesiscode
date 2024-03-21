import pickle as pkl
import numpy as np
from itertools import chain

from src.models.train_emos import train_emos, train_and_test_emos
from src.visualization.pit import make_cpit_diagram_emos, make_cpit_hist_emos 
from src.visualization.brier_score import brier_skill_plot, brier_plot
from src.models.get_data import get_tensors, get_normalized_tensor
from src.models.emos import EMOS
from src.visualization.scoring_tables import make_table
from src.visualization.twcrpss_plot import make_twcrpss_plot
from src.models.emos import EMOS
from src.visualization.reliability_diagram import make_reliability_diagram, make_reliability_diagram_sklearn
from src.training.training import load_model
from src.visualization.plot_forecasts import plot_forecast_cdf, plot_forecast_pdf, plot_weight_mixture

folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/'

base_model = load_model(folder + 'trunc_normal/tn_crps_.pkl')
print(base_model)

ml_tn_gev_M13_STD1 = load_model(folder + 'mixture_linear/mixturelinear_tn_gev_twcrps_mean13.0_std1.0_epochs600.pkl')

my_dict = {"ml_tn_gev_M13_STD1": ml_tn_gev_M13_STD1}

test_fold = 3
ignore = ['229', '285', '323']
X_test, y_test, variances_test = get_tensors(base_model.neighbourhood_size, base_model.feature_names, test_fold, ignore)
X_test = (X_test - base_model.feature_mean) / base_model.feature_std

values = np.linspace(0, 20, 200)
ylim = [-0.3, 0.3]
ylim = None
# brier_skill_plot(base_model, my_dict, X_test, y_test, variances_test, values, ylim=ylim)

threshold = 19
y_ = y_test[1802:1804]
variances_ = variances_test[1802:1804]
X_ = X_test[1802:1804,:]
# base_model.loss_Brier_score(X_test, y_test, variances_test, threshold)
#ml_tn_gev_M13_STD1.loss_Brier_score(X_test, y_test, variances_test, threshold)
print(ml_tn_gev_M13_STD1.loss_Brier_score(X_, y_, variances_, threshold))