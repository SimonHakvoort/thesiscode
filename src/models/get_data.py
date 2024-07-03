import os
import pickle as pkl
from typing import Tuple
import tensorflow as tf
from src.loading_data.forecast import Forecast

# retrieve the forecasts from the pickle files and returns them as a list
def get_folds():
    fold0 = get_fold_i(0)
    fold1 = get_fold_i(1)
    fold2 = get_fold_i(2)
    fold3 = get_fold_i(3)

    return fold0, fold1, fold2, fold3

def get_fold_i(i: int) -> list[Forecast]:
    """
    Load the forecasts from fold i from the pickle files and return them as a list.

    Args:
        i (int): the fold that needs to get loaded.

    Returns:
        A list of Forecast objects for fold i.
    """
    foldi = []

    for file in os.listdir(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/'):
        if file.endswith('.pkl'):
            with open(f'/net/pc200239/nobackup/users/hakvoort/fold{i}data/' + file, 'rb') as f:
                forecast = pkl.load(f)
                foldi.append(forecast)
    
    return foldi

def get_station_info() -> dict:
    """
    Loads station_info.pkl, which contains a dictionary with all the station information

    Arguments:
        None

    Returns:
        dict: dictionary with keys the station numbers and values dictionaries with keys 'lat' and 'lon' and values the latitude and longitude of the station.
    """
    with open('/net/pc200239/nobackup/users/hakvoort/station_info.pkl', 'rb') as f:
        station_info = pkl.load(f)
    return station_info

def get_tensors(neighbourhood_size: int, parameter_names: list, fold: int, ignore: list = []):
    """
    This function generates tensors 'X' and 'y' from the given parameters and fold.

    Parameters:
        neighbourhood_size (int): The size of the neighbourhood to consider.
        parameter_names (list): A list of parameter names to include in the tensor.
        fold (int): The fold to use for generating the samples.
        ignore (list, optional): A list of parameters to ignore. Defaults to an empty list.

    Returns:
        tuple: A tuple containing the tensor 'X' and the target tensor 'y'. If 'spatial_variance'. 
            is included in parameter_names, it is added to 'X' and removed from parameter_names.
    """
    fold = get_fold_i(fold)
    station_info = get_station_info()
    X_list = []
    y_list = []

    for forecast in fold:
        if forecast.has_observations():
            X, y = forecast.generate_all_samples(station_info, parameter_names, station_ignore=ignore, neighbourhood_size=neighbourhood_size)
            X_list.append(X)
            y_list.append(y)
    
    X = tf.concat(X_list, axis=0)
    y = tf.concat(y_list, axis=0)
        
    return X, y

def get_normalized_tensor(neighbourhood_size, parameter_names, folds, ignore = [], normalize_wind = False):
    """
    This function generates a normalized tensor from the given parameters.

    Parameters:
        neighbourhood_size (int): The size of the neighbourhood to consider.
        parameter_names (list): A list of parameter names to include in the tensor.
        folds (list): A list of folds to use for cross-validation.
        ignore (list, optional): A list of parameters to ignore. Defaults to an empty list.
        normalize_wind (bool, optional): If True, normalizes the wind parameter. Defaults to False.

    Returns:
        dict: A dictionary containing the normalized tensor 'X', the target tensor 'y', 
            the mean and standard deviation used for normalization, and the feature names.
    """
    X_list = []
    y_list = []

    for fold in folds:
        X, y = get_tensors(neighbourhood_size, parameter_names, fold, ignore)
        X_list.append(X)
        y_list.append(y)

    X = tf.concat(X_list, axis=0)
    y = tf.concat(y_list, axis=0)


    mean = tf.reduce_mean(X, axis=0)
    std = tf.math.reduce_std(X, axis=0)
    
    mean = tf.Variable(mean)
    std = tf.Variable(std)

    if not normalize_wind:
        mean[0].assign(tf.constant(0.0))
        std[0].assign(tf.constant(1.0))

    # check if spatial_variance is in the parameter_names
    if 'spatial_variance' in parameter_names:
        # find the index in parameter_names
        index = parameter_names.index('spatial_variance')
        mean[index].assign(tf.constant(0.0))
        std[index].assign(tf.constant(1.0))

    X = (X - mean) / std

    feature_names = parameter_names.copy()

    output_dict = {'X': X, 'y': y, 'mean': mean, 'std': std, 'features_names': feature_names}

    return output_dict
    
def sort_tensor(X: tf.Tensor, y: tf.Tensor, variance = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Sorts the tensors based on the value of y. The tensors X and variance have the same order.
    """
    order = tf.argsort(y, axis=0, direction='DESCENDING')
    X = tf.gather(X, order)
    y = tf.gather(y, order)

    if variance is not None:
        variance = tf.gather(variance, order)

    return X, y, variance
    