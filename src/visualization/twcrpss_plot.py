import matplotlib.pyplot as plt
import numpy as np

from src.climatology.climatology import Climatology
from src.models.emos import EMOS
from src.neural_networks.nn_forecast import NNForecast

def comp_twcrpss(model_ref, model, X, y, threshold, sample_size = 1000):
    return 1 - model.twCRPS(X, y, threshold, sample_size).numpy() / model_ref.twCRPS(X, y, threshold, sample_size).numpy()

def comp_twcrps_tf(model, data, thresholds, sample_size = 1000):
    scores = []
    if type(model) == EMOS:
        for threshold in thresholds:
            scores.append(model.twCRPS_tfdataset(data, threshold, sample_size).numpy())
    elif type(model) == NNForecast:
        scores = model.twCRPS(data, thresholds, sample_size)
    elif isinstance(model, Climatology):
        scores = model.get_twCRPS(data, thresholds)
    else:
        raise ValueError('Model type not recognized')
    return scores

def make_twcrpss_plot(base_model, model_dict, X, y, thresholds, ylim = None, sample_size = 1000):
    """
    Makes a plot of the twCRPSS for different thresholds for the models in model_dict, compared to base_model

    Arguments:
    - base_model: the model to compare to
    - model_dict: a dictionary of models to compare
    - X: the input data
    - y: the output data
    - thresholds: the thresholds to compare the models at
    - ylim: the limits of the y-axis
    - sample_size: the number of samples to use for the twCRPSS calculation
    """
    for model_name, model in model_dict.items():
        scores = [comp_twcrpss(base_model, model, X, y, threshold, sample_size) for threshold in thresholds]
        #check if scores contains nan
        if np.isnan(scores).any():
            print("scores contain nan")
        plt.plot(thresholds, scores, label = model_name)

    # plot horizontal black striped line at y=0
    plt.plot([thresholds[0], thresholds[-1]], [0, 0], color="black", linestyle="dashed")

    plt.xlabel('Threshold (m/s)')
    plt.ylabel('twCRPSS')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(thresholds[0], thresholds[-1])
    plt.legend()
    plt.show()


def make_twcrpss_plot_tf(base_model, model_dict, data, thresholds, ylim = None, sample_size = 1000):
    base_model_twcrps = comp_twcrps_tf(base_model, data, thresholds, sample_size)
    for model_name, model in model_dict.items():
        scores = comp_twcrps_tf(model, data, thresholds, sample_size)
        scores = 1 - np.array(scores) / np.array(base_model_twcrps)
        plt.plot(thresholds, scores, label = model_name)

    # plot horizontal black striped line at y=0
    plt.plot([thresholds[0], thresholds[-1]], [0, 0], color="black", linestyle="dashed")
    
    plt.xlabel('Threshold (m/s)')
    plt.ylabel('twCRPSS')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(thresholds[0], thresholds[-1])
    plt.legend()
    plt.show()


def make_twcrps_plot(model_dict, X, y, threshold, ylim = None, sample_size = 1000, base_model = None):
    """
    Makes a plot of the twCRPS for different models

    Arguments:
    - model_dict: a dictionary of models
    - X: the input data
    - y: the output data
    - variances: the variances of the output data
    - threshold: the threshold to compare the models at
    - ylim: the limits of the y-axis
    - sample_size: the number of samples to use for the twCRPS calculation
    - base_model: the base model
    """
    for model_name, model in model_dict.items():
        scores = []
        for value in threshold:
            scores.append(model.twCRPS(X, y, value, sample_size).numpy())
        plt.plot(threshold, scores, label = model_name)

    if base_model is not None:
        scores = []
        for value in threshold:
            scores.append(base_model.twCRPS(X, y, value, sample_size).numpy())
        plt.plot(threshold, scores, label = "Base model", color = "black")
    plt.xlabel('Threshold (m/s)')
    plt.ylabel('twCRPS')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(threshold[0], threshold[-1])
    plt.legend()
    plt.show()