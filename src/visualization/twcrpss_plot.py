import matplotlib.pyplot as plt
import numpy as np

def comp_twcrpss(model_ref, model, X, y, variances, threshold, sample_size = 1000):
    return 1 - model.loss_twCRPS_indicator_sample(X, y, variances, threshold, sample_size).numpy() / model_ref.loss_twCRPS_indicator_sample(X, y, variances, threshold, sample_size).numpy()


def make_twcrpss_plot(base_model, model_dict, X, y, variances, thresholds, ylim = None, sample_size = 1000):
    """
    Makes a plot of the twCRPSS for different thresholds for the models in model_dict, compared to base_model

    Arguments:
    - base_model: the model to compare to
    - model_dict: a dictionary of models to compare
    - X: the input data
    - y: the output data
    - variances: the variances of the output data
    - thresholds: the thresholds to compare the models at
    - ylim: the limits of the y-axis
    - sample_size: the number of samples to use for the twCRPSS calculation
    """
    for model_name, model in model_dict.items():
        scores = [comp_twcrpss(base_model, model, X, y, variances, threshold, sample_size) for threshold in thresholds]
        plt.plot(thresholds, scores, label = model_name)

    # plot horizontal black striped line at y=0
    plt.plot([thresholds[0], thresholds[-1]], [0, 0], color="black", linestyle="dashed")

    plt.xlabel('Threshold (m/s)')
    plt.ylabel('twCRPSS')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()
