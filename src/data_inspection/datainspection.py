import os
import numpy as np
import pandas as pd
import pygrib
import pickle as pkl
from src.models.get_data import get_folds, get_station_info
from itertools import chain


# grb = '/net/pc230023/nobackup/users/ameronge/windwinter/output/'
# file = 'HA40_N25_WINDFC_201711240000_02400_GB'

# grbs = pygrib.open(grb + file)

# y = grbs.select(indicatorOfParameter=1, level=0)

# x = 3

# windgusts code yellow: 75 km/h -> 20.83 m/s


station_info = get_station_info()

fold_test, fold1, fold2, fold3 = get_folds()
omit = ['229', '285', '235']

data_test = []
data_1 = []
data_2 = []
data_3 = []

variable = ['wind_speed']

for forecast in fold_test:
    if forecast.has_observations():
        X, y, variances = forecast.generate_all_samples(3, station_info, variable, omit)
        data_test.append(y.numpy())

for forecast in fold1:
    if forecast.has_observations():
        X, y, variances = forecast.generate_all_samples(3, station_info, variable, omit)
        data_1.append(y.numpy())

for forecast in fold2:
    if forecast.has_observations():
        X, y, variances = forecast.generate_all_samples(3, station_info, variable, omit)
        data_2.append(y.numpy())

for forecast in fold3:
    if forecast.has_observations():
        X, y, variances = forecast.generate_all_samples(3, station_info, variable, omit)
        data_3.append(y.numpy())




data_test = np.array(list(chain.from_iterable(data_test)))
data_1 = np.array(list(chain.from_iterable(data_1)))
data_2 = np.array(list(chain.from_iterable(data_2)))
data_3 = np.array(list(chain.from_iterable(data_3)))

percentiles = [90, 95, 99, 99.5]
#consider between 95 and 99
percentiles_test = np.percentile(data_test, percentiles)
percentiles_1 = np.percentile(data_1, percentiles)
percentiles_2 = np.percentile(data_2, percentiles)
percentiles_3 = np.percentile(data_3, percentiles)

print("Test: ", percentiles_test)
print("1: ", percentiles_1)
print("2: ", percentiles_2)
print("3: ", percentiles_3)

yellow_warning = 20.83
gust_factor = 1.5

threshold = yellow_warning / gust_factor

print("Test: ", np.sum(data_test > threshold))
print("1: ", np.sum(data_1 > threshold))
print("2: ", np.sum(data_2 > threshold))
print("3: ", np.sum(data_3 > threshold))

print("top 20 values in test: " , np.sort(data_test)[-20:])
