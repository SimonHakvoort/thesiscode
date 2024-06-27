import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
from src.models.emos import EMOS
from src.neural_networks.nn_forecast import NNForecast

def comp_twcrpss(model_ref, model, X, y, threshold, sample_size = 1000):
    return 1 - model.twCRPS(X, y, threshold, sample_size).numpy() / model_ref.twCRPS(X, y, threshold, sample_size).numpy()

### Should get deleted.
def comp_twcrps_tf(model, data, thresholds, sample_size = 1000):
    scores = []
    print("This is an old function!!!!")
    if type(model) == EMOS:
        scores = model.twCRPS(data, thresholds, sample_size)
    elif type(model) == NNForecast:
        scores = model.twCRPS(data, thresholds, sample_size)
    elif isinstance(model, Climatology):
        scores = model.twCRPS(data, thresholds, sample_size)
    else:
        raise ValueError('Model type not recognized')
    return scores




def make_twcrpss_plot_tf(base_model, model_dict: dict, data: tf.data.Dataset, values: np.ndarray, ylim: tuple = None, sample_size: int = 1000, base_model_name: str = 'Reference Model'):
    """
    Plots the thesdhold-weighted CRPS (twCRPS) for the models, which is a dict for the numbers in values. 
    Evaluates the performance for a single batch in data.

    Arguments:
        basemodel: reference model
        models (dict): models to compare to basemodel
        data (tf.data.Dataset): data to compute twCRPS.
        values (np.array): values to compute the twCRPSS.
        ylim (tuple, optional): tuple specifying the range of the y-axis.
        title (str, optional: Title for the plot.
        name_base_model (str, optional): name for the reference model in the legend.

    Returns:
        None
    """
    base_model_twcrps = base_model.twCRPS(data, values, sample_size)
    for model_name, model in model_dict.items():
        scores = model.twCRPS(data, values, sample_size)
        scores = 1 - np.array(scores) / np.array(base_model_twcrps)
        plt.plot(values, scores, label = model_name)

    # plot horizontal black striped line at y=0
    plt.plot([values[0], values[-1]], [0, 0], color="black", linestyle="dashed", label=base_model_name)
    
    plt.xlabel('Threshold (m/s)')
    plt.ylabel('twCRPSS')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlim(values[0], values[-1])
    plt.legend()
    plt.show()


