import os
import pickle as pkl

import numpy as np


from src.models.emos import EMOS
from src.models.get_data import get_normalized_tensor, get_tensors
    

def train_emos(neighbourhood_size, parameter_names, epochs, folds, setup, ignore = []):
    data = get_normalized_tensor(neighbourhood_size, parameter_names, folds, ignore = ignore)

    X = data["X"]
    y = data["y"]

    setup["all_features"] = data["features_names"]
    setup["feature_mean"] = data["mean"]
    setup["feature_std"] = data["std"]
    setup["features"] = parameter_names
    setup["neighbourhood_size"] = neighbourhood_size

    subset_size = None
    if "subset_size" in setup:
        subset_size = setup["subset_size"]

    printing = False
    if "printing" in setup:
        printing = setup["printing"]

    emos = EMOS(setup)

    emos.fit(X, y, epochs, printing = printing, subset_size = subset_size)

    return emos

def train_and_test_emos(neighbourhood_size, parameter_names, epochs, train_folds, test_folds, setup):
    data = get_normalized_tensor(neighbourhood_size, parameter_names, train_folds)

    X = data["X"]
    y = data["y"]
    variances = data["variances"]

    setup["num_features"] = len(parameter_names)
    setup["feature_mean"] = data["mean"]
    setup["feature_std"] = data["std"]
    setup["features"] = parameter_names
    setup["neighbourhood_size"] = neighbourhood_size

    emos = EMOS(setup)
    emos.fit(X, y, variances, epochs)

    X_test, y_test, variances_test = get_tensors(neighbourhood_size, parameter_names, test_folds)


    X_test = (X_test - data["mean"]) / data["std"]
    emos.samples = 1000
    return {"model" : emos, "test_loss" : emos.loss(X_test, y_test, variances_test)}

def train_and_save_emos(neighbourhood_size, parameter_names, epochs, folds, setup):
    emos = train_emos(neighbourhood_size, parameter_names, epochs, folds, setup)
    mydict = emos.to_dict()
    with open('/net/pc200239/nobackup/users/hakvoort/models/emos_{0}_{1}.pkl'.format(setup['loss'], setup['forecast_distribution']), 'wb') as f:
        pkl.dump(mydict, f)
    
    print("Model saved")
    


