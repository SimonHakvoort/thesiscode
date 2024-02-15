import os
import numpy as np
import pandas as pd
import pygrib
import pickle as pkl
from loadforecasts import get_folds, get_station_info

# grb = '/net/pc230023/nobackup/users/ameronge/windwinter/output/'
# file = 'HA40_N25_WINDFC_201711240000_02400_GB'

# grbs = pygrib.open(grb + file)

# y = grbs.select(indicatorOfParameter=1, level=0)

# x = 3

station_info = get_station_info()

fold_test, fold1, fold2, fold3 = get_folds()
training_folds = [fold1, fold2, fold3]
omit = ['229', '285', '235']

training_observations = np.array()
testing_observations = np.array()

for fold in training_folds:
    for forecast in fold:
        X, y, variances = forecast.generate_all_samples(3, station_info, omit)
        training_observations.append(y)

for forecast in fold_test:
    X, y, variances = forecast.generate_all_samples(3, station_info, omit)
    testing_observations.append(y)

percentile = 95

print("The 95th percentile of the training observations is: ", np.percentile(training_observations, percentile))
print("The 95th percentile of the testing observations is: ", np.percentile(testing_observations, percentile))


