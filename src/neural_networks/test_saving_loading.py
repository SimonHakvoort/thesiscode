import pickle
from src.neural_networks.get_data import get_tf_data, normalize_1d_features, normalize_1d_features_with_mean_std, stack_1d_features
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

train_data = train_data.shuffle(len(train_data))

train_data = train_data.batch(32)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)


filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/test2'

model = NNForecast.my_load(filepath, train_data)


fold = 3
test_data = get_tf_data([fold], features_names_dict, ignore=ignore)

test_data = test_data.map(lambda x, y: stack_1d_features(x, y))

test_data = normalize_1d_features_with_mean_std(test_data, model.features_1d_mean, model.features_1d_std)

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

print("The CRPS is: ", model.CRPS(test_data, 1000).numpy())

print("The twCRPS is: ", model.twCRPS(test_data, [10, 12, 14], 1000))

print(model.loss_function.__name__)
