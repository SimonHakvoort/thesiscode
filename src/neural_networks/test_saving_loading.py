from neural_networks.get_data import get_tf_data, normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf
from src.neural_networks.nn_model import NNModel

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

train_data = get_tf_data([1,2], features_names_dict, ignore=ignore)

train_data = train_data.map(lambda x, y: stack_1d_features(x, y))

train_data, mean, std = normalize_1d_features(train_data)





filepath = '/net/pc200239/nobackup/users/hakvoort/models/non_conv_nn/test'

model = NNForecast.my_load(filepath)

fold = 3
test_data = get_tf_data([fold], features_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data = normalize_1d_features_with_mean_std(test_data, mean, std)

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

print(model.CRPS(test_data, 1000))

print(model.twCRPS(test_data, 1000, 12))