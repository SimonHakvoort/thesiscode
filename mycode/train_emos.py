import os
import pickle as pkl
import tensorflow as tf

from emos import EMOS

datafolder = '/net/pc200239/nobackup/users/hakvoort/fold1data/'
validationfolder = '/net/pc200239/nobackup/users/hakvoort/fold1data/'

forecasts1 = []
for file in os.listdir(datafolder):
    if file.endswith('.pkl'):
        with open(datafolder + file, 'rb') as f:
            forecast = pkl.load(f)
            forecasts1.append(forecast)

forecasts_val = []
for file in os.listdir(validationfolder):
    if file.endswith('.pkl'):
        with open(validationfolder + file, 'rb') as f:
            forecast = pkl.load(f)
            forecasts_val.append(forecast)

station_info = pkl.load(open('/net/pc200239/nobackup/users/hakvoort/station_info.pkl', 'rb'))

neighbourhood_size = 3

X_list = []
y_list = []
variances_list = []
for forecast in forecasts1:
    X, y, variances = forecast.generate_all_samples(neighbourhood_size, station_info)
    if X.shape != (0,):
        X_list.append(X)
        y_list.append(y)
        variances_list.append(variances)

X_1 = tf.concat(X_list, axis=0)
y_1 = tf.concat(y_list, axis=0)
variances_1 = tf.concat(variances_list, axis=0)

X_list = []
y_list = []
variances_list = []
for forecast in forecasts_val:
    if forecast.has_observations():
        X, y, variances = forecast.generate_all_samples(neighbourhood_size, station_info)
        X_list.append(X)
        y_list.append(y)
        variances_list.append(variances)
                

X_val = tf.concat(X_list, axis=0)
y_val = tf.concat(y_list, axis=0)
variances_val = tf.concat(variances_list, axis=0)

emos = EMOS(2)

for _ in range(0, 500):
    emos.fit(X_1, y_1, variances_1, 10)
    validation_loss = emos.loss(X_val, y_val, variances_val)
    print("Validation loss: ", validation_loss)

a, b, c, d = emos.get_params()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)