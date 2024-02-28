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

def get_tensors(neighbourhood_size, parameter_names, fold, ignore = []):
    fold = get_fold_i(fold)
    station_info = get_station_info()
    X_list = []
    y_list = []
    variances_list = []
    for forecast in fold:
        if forecast.has_observations():
            X, y, variances = forecast.generate_all_samples(neighbourhood_size, station_info, parameter_names, ignore)
            X_list.append(X)
            y_list.append(y)
            variances_list.append(variances)
    
    X = tf.concat(X_list, axis=0)
    y = tf.concat(y_list, axis=0)
    variances = tf.concat(variances_list, axis=0)
    return X, y, variances

def get_normalized_tensor(neighbourhood_size, parameter_names, folds, ignore = []):
    """
    Returns a dictionary containing the normalized X (except the first row), y and variances of the neighberhoods in the folds, except for the stations from ignore.
    In the dictionary this can be accessed with the keys 'X', 'y' and 'variances'. The mean and standard deviation of the features are also included in the dictionary, 
    which can be accessed with the keys 'mean' and 'std'.

    Args:
    - neighbourhood_size: int
    - parameter_names: list of strings
    - folds: list of ints
    - ignore: list of strings

    Returns:
    - dict: dictionary
    """
    X_list = []
    y_list = []
    variances_list = []
    for fold in folds:
        X, y, variances = get_tensors(neighbourhood_size, parameter_names, fold, ignore)
        X_list.append(X)
        y_list.append(y)
        variances_list.append(variances)

    X = tf.concat(X_list, axis=0)
    y = tf.concat(y_list, axis=0)
    variances = tf.concat(variances_list, axis=0)    

    mean = tf.reduce_mean(X, axis=0)
    std = tf.math.reduce_std(X, axis=0)
    
    mean = tf.Variable(mean)
    std = tf.Variable(std)

    mean[0].assign(tf.constant(0.0))
    std[0].assign(tf.constant(1.0))

    X = (X - mean) / std
    output_dict = {'X': X, 'y': y, 'variances': variances, 'mean': mean, 'std': std}
    return output_dict
    
def sort_tensor(X, y, variance):
    order = tf.argsort(y, axis=0, direction='DESCENDING')
    X = tf.gather(X, order)
    y = tf.gather(y, order)
    variance = tf.gather(variance, order)

    return X, y, variance
    