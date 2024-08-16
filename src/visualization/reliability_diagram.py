import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import tensorflow as tf
from sklearn.calibration import calibration_curve

from src.linreg_emos.emos import BaseForecastModel


def make_reliability_and_sharpness(dict: dict[str: BaseForecastModel], data: tf.data.Dataset, t: float, n_subset: int = 10, base_model: BaseForecastModel = None, base_model_name = 'Base Model') -> None:
    """
    Makes a reliability diagram and a sharpness diagram for the models in dict, for wind speeds of value t.

    Arguments:
        dict (dict[str: BaseForecastModel]): a dictionary containing the models.
        data (tf.data.Dataset): the data for which we make the reliability and sharpness diagram.
        t (float): the threshold at which we make the diagram.
        n_subset (int): the number of bins we use to split the data.
        base_model (BaseForecastModel): an optional model that will also be shown in the diagram.
        base_model_name (str): the name of the base_model.

    Returns: 
        None
    """
    fig = plt.figure(figsize=(8, 8))  # Create a figure
    gs = gridspec.GridSpec(4, 1)  # Create a GridSpec with 3 rows and 1 column

    # Create the subplots with different heights
    axs = [plt.subplot(gs[:3]), plt.subplot(gs[3])]

    #data is a tf.data.Dataset
    X, y = next(iter(data))

    y_true = y > t
    cdfs = {}

    for name, model in dict.items():
        distribution, observations = model.get_prob_distribution(data)
        probs = 1.0 - distribution.cdf(t)
        probs = np.clip(probs, 0, 1)
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_subset)
        axs[0].plot(prob_pred, prob_true, 'o-', label = name)
        cdfs[name] = probs

    if base_model is not None:
        distribution, observations = base_model.get_prob_distribution(data)
        cdf_values = np.clip(1 - distribution.cdf(t), 0, 1)
        prob_true, prob_pred = calibration_curve(y_true, cdf_values, n_bins=n_subset)
        axs[0].plot(prob_pred, prob_true, 'o-', label=base_model_name)
        cdfs["Base model"] = cdf_values

    axs[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
    axs[0].set_xlabel("Mean predicted probability")
    axs[0].set_ylabel("Fraction of positives")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[0].legend()

    subset_values = np.linspace(0, 1, n_subset)
    for name, y_prob in cdfs.items():
        counts, bin_edges = np.histogram(y_prob, bins = subset_values)
        counts = np.append(counts, 0)
        axs[1].step(subset_values, counts / len(y) * 100, where='post', label = name)

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Count (%)")
    axs[1].set_xlim(0, 1)
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()






