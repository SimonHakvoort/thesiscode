from src.models.get_data import get_tensors
from src.training.training import load_model
from src.visualization.reliability_diagram import make_reliability_diagram_sklearn

folder = '/net/pc200239/nobackup/users/hakvoort/models/emos/'

base_model = load_model(folder + 'trunc_normal/tn_crps_.pkl')
tn = load_model(folder + 'trunc_normal/tn_twcrps_mean13.0_std2.0_epochs600.pkl')
gev = load_model(folder + 'gev/gev_twcrps_mean13.0_std2.0_epochs600.pkl')

test_fold = 3
ignore = ['229', '285', '323']
X_test, y_test, variances_test = get_tensors(tn.neighbourhood_size, tn.all_features, test_fold, ignore)
X_test = (X_test - tn.feature_mean) / tn.feature_std

mydict = {'base_model': base_model, 'tn': tn, 'gev': gev}
t = 8

make_reliability_diagram_sklearn(mydict, X_test, y_test, variances_test, t, 11)
