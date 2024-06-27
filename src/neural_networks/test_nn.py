
import time
import keras
import numpy as np
from src.neural_networks.get_data import get_fold_is, load_cv_data, load_train_test_data, make_importance_sampling_dataset, normalize_1d_features, normalize_1d_features_with_mean_std, save_cv_data, stack_1d_features, get_tf_data
from src.neural_networks.nn_model import NNModel
from src.models.get_data import get_tensors
from src.neural_networks.nn_forecast import NNForecast
import tensorflow as tf
import pickle
import os

from src.visualization.pit import comp_pit_score_tf

features_names = ['wind_speed', 'press', 'kinetic', 'humid', 'geopot']

features_names_dict = {name: 1 for name in features_names}

features_names_dict['wind_speed'] = 15

ignore = ['229', '285', '323']

bounds = {7.5: 1, 9: 3, 12: 4, 15: 9, 100: 15}

train_data, test_data, data_info = load_cv_data(3, features_names_dict)

original_data_size = train_data.cardinality().numpy()

batch_size = 32

train_data = make_importance_sampling_dataset(train_data, bounds)

train_data = train_data.cache()

# dataset_length = [i for i,_ in enumerate(train_data)][-1] + 1

dataset_length = 28595

train_data = train_data.shuffle(dataset_length)

steps_per_epoch = original_data_size // batch_size

print(steps_per_epoch)

print(dataset_length)

train_data = train_data.batch(batch_size)

train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

train_data = train_data.repeat()

test_data = test_data.batch(len(test_data))

test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)






forecast_distribution = 'distr_trunc_normal'
distribution_1 = 'distr_trunc_normal'
distribution_2 = 'distr_log_normal'

loss_function = 'loss_twCRPS_sample'
chain_function = 'chain_function_normal_cdf_plus_constant'
chain_function_mean = 9
chain_function_std = 0.25
chain_function_constant = 0.01

optimizer = 'adam'
learning_rate = 0.0005


dense_l1_regularization = 0.000
dense_l2_regularization = 0.0003
hidden_units_list = [100, 100, 100]
add_nwp_forecast = True
conv_7x7_units = 4
conv_5x5_units = 4
conv_3x3_units = 4
add_wind_conv = True

metrics = ['twCRPS_12']# ['twCRPS_10', 'twCRPS_12', 'twCRPS_15']
metrics = None
saving = True

epochs = 200

filepath = '/net/pc200239/nobackup/users/hakvoort/models/conv_nn/'

if loss_function == 'loss_twCRPS_sample':
    name = 'twCRPS'
    name += '_mean_' + str(chain_function_mean)
    name += '_std_' + str(chain_function_std)
    name += '_constant_' + str(chain_function_constant)
elif loss_function == 'loss_CRPS_sample':
    name = 'CRPS'

filepath += name + '_'

if forecast_distribution == 'distr_mixture':
    filepath += 'mixture_'
elif forecast_distribution == 'distr_trunc_normal':
    filepath += 'trunc_normal_'
elif forecast_distribution == 'distr_log_normal':
    filepath += 'log_normal_'

filepath += 'epochs_' + str(epochs) 



filepath += '_is_3'


# make a folder
if saving:
    os.makedirs(filepath, exist_ok=True)

setup_distribution = {
    'forecast_distribution': forecast_distribution,
    'distribution_1': distribution_1,
    'distribution_2': distribution_2,
}

setup_nn_architecture = {
    'hidden_units_list': hidden_units_list,
    'dense_l1_regularization': dense_l1_regularization,
    'dense_l2_regularization': dense_l2_regularization,
    'add_nwp_forecast': add_nwp_forecast,

    'conv_7x7_units': conv_7x7_units,
    'conv_5x5_units': conv_5x5_units,
    'conv_3x3_units': conv_3x3_units,
}

setup_loss = {
    'loss_function': loss_function,
    'chain_function': chain_function,
    'chain_function_mean': chain_function_mean,
    'chain_function_std': chain_function_std,
    'chain_function_constant': chain_function_constant,
}

setup_optimizer = {
    'optimizer': optimizer,
    'learning_rate': learning_rate,
}

setup = {
    'setup_distribution': setup_distribution,
    'features_names': features_names,
    'setup_loss': setup_loss,
    'setup_optimizer': setup_optimizer,
    'sample_size': 250,
    'setup_nn_architecture': setup_nn_architecture,

    'add_wind_conv': add_wind_conv,

    'features_1d_mean': data_info['features_1d_mean'],
    'features_1d_std': data_info['features_1d_std'],
    'metrics': metrics,
}

if saving:
    with open(filepath + '/attributes', 'wb') as f:
        pickle.dump(setup, f)

nn = NNForecast(**setup)

#start the time
time_start = time.time()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = nn.fit(train_data, epochs=epochs, validation_data=test_data , early_stopping=early_stopping, steps_per_epoch=steps_per_epoch)

best_epoch = early_stopping.stopped_epoch - early_stopping.patience

print(f'Best epoch: {best_epoch}')

#end the time
time_end = time.time()

print("Time: ", time_end - time_start)

print(nn.CRPS(test_data, 10000).numpy())

print(nn.twCRPS(test_data, [12], 10000)[0].numpy())

values = np.linspace(0, 20, 40)

brierscores = nn.Brier_Score(test_data, values)


# X, y = next(iter(test_data))

# shape = nn.get_gev_shape(X)

if saving:
    nn.save_weights(filepath)
    print("Model saved")

# save the history
if saving:
    with open(filepath + '/history.pickle', 'wb') as f:
        pickle.dump(history.history, f)
        print("History saved")





