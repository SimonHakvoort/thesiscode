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
    