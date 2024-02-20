import os
import pickle as pkl
import tensorflow as tf

from src.models.emos import EMOS
from src.models.get_data import get_tensors



neighbourhood_size = 5

# possible parameters: 'wind_speed', 'press', 'kinetic', 'humid', 'geopot'
parameter_names = ['wind_speed', 'kinetic', 'humid']

X_1, y_1, variances_1 = get_tensors(neighbourhood_size, parameter_names, 1)
X_val, y_val, variances_val = get_tensors(neighbourhood_size, parameter_names, 0)

emos_ll = EMOS(len(parameter_names), loss = "log_likelihood")
emos_crps = EMOS(len(parameter_names), loss = "CRPS_sample")

emos_crps.fit(X_1, y_1, variances_1, 500)
a, b, c, d = emos_crps.get_params()
print("a: ", a)
print("b: ", b)
print("c: ", c)
print("d: ", d)

val_loss = emos_crps.loss(X_val, y_val, variances_val)
print("Validation loss: ", val_loss)