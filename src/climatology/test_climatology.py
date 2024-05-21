

from src.climatology.climatology import Climatology
from src.neural_networks.get_data import load_cv_data


features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

folds = 3

folder = '/net/pc200239/nobackup/users/hakvoort/models/climatology/climatology_cv_3'

train_data, test_data, data_info = load_cv_data(3, features_names_dict)

original_data = list(train_data.as_numpy_iterator())

test_data = test_data.batch(len(test_data))


climatology = Climatology.load(folder)

values = [0, 5, 10, 15, 20]

bier_scores = climatology.get_Brier_scores(test_data, values)

twcrps = climatology.get_twCRPS(test_data, values, 10000)

print(twcrps)