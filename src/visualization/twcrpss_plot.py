import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
from src.linreg_emos.emos import BaseForecastModel, LinearEMOS
from src.cnn_emos.nn_forecast import CNNEMOS

def comp_twcrpss(model_ref, model, X, y, threshold, sample_size = 1000):
    return 1 - model.twCRPS(X, y, threshold, sample_size).numpy() / model_ref.twCRPS(X, y, threshold, sample_size).numpy()


def make_twcrpss_plot_tf(base_model: BaseForecastModel, 
                         model_dict: dict[str, BaseForecastModel], 
                         data: tf.data.Dataset, 
                         values: np.ndarray, 
                         ylim: tuple = None, 
                         sample_size: int = 1000, 
                         base_model_name: str = 'Reference Model'):
    """
    Plots the theshold-weighted CRPS (twCRPS) for the models (stored in a dict) for the thresholds in values. 
    Evaluates the performance for a single batch in data.

    Arguments:
        basemodel (BaseForecastModel): reference model
        models (dict[str, BaseForecastModel]): models to compare to basemodel
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
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


