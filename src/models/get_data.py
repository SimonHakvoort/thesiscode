import os
import pickle as pkl
import tensorflow as tf

# retrieve the forecasts from the pickle files and returns them as a list
def get_folds():
    fold0 = get_fold_i(0)
    fold1 = get_fold_i(1)
    fold2 = get_fold_i(2)
    fold3 = get_fold_i(3)

    return fold0, fold1, fold2, fold3

def get_fold_i(i):
    foldi = []

    for file in os.listdir(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/'):
        if file.endswith('.pkl'):
            with open(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/' + file, 'rb') as f:
                forecast = pkl.load(f)
                foldi.append(forecast)
    
    return foldi

def get_station_info():
    with open('/net/pc200239/nobackup/users/hakvoort/station_info.pkl', 'rb') as f:
        station_info = pkl.load(f)
    return station_info

def get_tensors(neighberhood_size, parameter_names, fold):
    fold = get_fold_i(fold)
    station_info = get_station_info()
    X_list = []
    y_list = []
    variances_list = []
    for forecast in fold:
        if forecast.has_observations():
            X, y, variances = forecast.generate_all_samples(neighberhood_size, station_info, parameter_names)
            X_list.append(X)
            y_list.append(y)
            variances_list.append(variances)
    
    X = tf.concat(X_list, axis=0)
    y = tf.concat(y_list, axis=0)
    variances = tf.concat(variances_list, axis=0)
    return X, y, variances