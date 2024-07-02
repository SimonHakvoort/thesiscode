import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.gridspec as gridspec
import tensorflow as tf


def make_reliability_and_refinement_diagram(emos_dict, X, y, variances, t, n_subset = 11):
    subset_values = np.linspace(0, 1, n_subset)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create a figure with two subplots

    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t).numpy()

        empirical_probabilities = np.zeros(n_subset - 1)
        average_cdf_values = np.zeros(n_subset - 1)
        counts = np.zeros(n_subset - 1)
        for i in range(0, len(cdf_values)):
            for j in range(0, n_subset - 1):
                if cdf_values[i] >= subset_values[j] and cdf_values[i] < subset_values[j + 1]:
                    counts[j] += 1
                    average_cdf_values[j] += cdf_values[i]
                    if y[i] < t:
                        empirical_probabilities[j] += 1

        axs[0].plot(average_cdf_values / counts, empirical_probabilities / counts, 'o-', label = name)
        
        # #in ax[1] we will plot the refinement diagram, which contains the counts for each bin
        axs[1].plot(subset_values[0:-1], counts / len(cdf_values), label = name)

        



    axs[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
    axs[0].set_xlabel("Forecast probability")
    axs[0].set_ylabel("Empirical probability")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Count")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def compute_empirical_and_forecast_probabilities(cdf_values, subset_values, y, t):
    empirical_probabilities = np.zeros(len(subset_values) - 1)
    average_cdf_values = np.zeros(len(subset_values) - 1)
    counts = np.zeros(len(subset_values) - 1)
    for i in range(0, len(cdf_values)):
        for j in range(0, len(subset_values) - 1):
            if cdf_values[i] >= subset_values[j] and cdf_values[i] < subset_values[j + 1]:
                counts[j] += 1
                average_cdf_values[j] += cdf_values[i]
                if y[i] < t:
                    empirical_probabilities[j] += 1
    return empirical_probabilities, average_cdf_values, counts


def make_reliability_diagram(emos_dict, X, y, variances, t, n_subset = 11, base_model = None):
    """
    Makes a reliability diagram for each EMOS model in the dictionary, for the given data and threshold t.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - t: real valued number greater than 0
    - n_subset: integer, number of subsets to divide the forecast probabilities in
    - base_model: EMOS model, optional

    Returns:
    - None
    """
    subset_values = np.linspace(0, 1, n_subset)

    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t).numpy()

        empirical_probabilities, average_cdf_values, counts = compute_empirical_and_forecast_probabilities(cdf_values, subset_values, y, t)

        plt.plot(average_cdf_values / counts, empirical_probabilities / counts, 'o-', label = name)
    
    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X, variances)
        cdf = distributions.cdf
        cdf_values = cdf(t).numpy()

        empirical_probabilities, average_cdf_values, counts = compute_empirical_and_forecast_probabilities(cdf_values, subset_values, y, t)

        plt.plot(average_cdf_values / counts, empirical_probabilities / counts, 'o-', label = "Base model")

    plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")
    plt.xlabel("Forecast probability")
    plt.ylabel("Empirical probability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def make_reliability_diagram_sklearn(emos_dict, X, y, t, n_subset = 10, base_model = None):
    y_true = y > t
    cdfs = {}
    for name, model in emos_dict.items():
        probs = 1.0 - model.forecast_distribution.comp_cdf(X, t)
        # if probs is greater than 1, set it to 1 or if it is less than 0, set it to 0
        probs = np.clip(probs, 0, 1)
        cdfs[name] = probs.squeeze()


    for name, y_prob in cdfs.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_subset)
        plt.plot(prob_pred, prob_true, 'o-', label = name)

    if base_model is not None:
        # distributions = base_model.forecast_distribution.get_distribution(X, variances)
        # cdf = distributions.cdf
        # cdf_values = cdf(t).numpy()
        cdf_values = np.clip(1 - base_model.forecast_distribution.comp_cdf(X, t), 0, 1).squeeze()
        prob_true, prob_pred = calibration_curve(y_true, 1 - cdf_values, n_bins=n_subset)
        plt.plot(prob_pred, prob_true, 'o-', label = "Base model")
    
    plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def make_sharpness_diagram(emos_dict, X, y, t, n_subset = 10, base_model = None):
    subset_values = np.linspace(0, 1, n_subset)
    cdfs = {}
    for name, model in emos_dict.items():
        probs = 1.0 - model.forecast_distribution.comp_cdf(X, t)
        # if probs is greater than 1, set it to 1 or if it is less than 0, set it to 0
        probs = np.clip(probs, 0, 1)
        cdfs[name] = probs.squeeze()

    if base_model is not None:
        cdfs["Base model"] = np.clip(1 - base_model.forecast_distribution.comp_cdf(X, t), 0, 1).squeeze()


    for name, y_prob in cdfs.items():
        counts, bin_edges = np.histogram(y_prob, bins = subset_values)
        counts = np.append(counts, 0)
        plt.step(subset_values, counts / len(y) * 100, where='post', label = name)

    plt.xlabel("Forecast probability")
    plt.ylabel("Count (%)")
    plt.xlim(0, 1)
    plt.legend()
    plt.show()


def make_reliability_and_sharpness(emos_dict, X, y, t, n_subset = 10, base_model = None):
    """
    Makes a reliability diagram and a sharpness diagram for each EMOS model in the dictionary, for the given data and threshold t.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - t: real valued number greater than 0
    - n_subset: integer, number of subsets to divide the forecast probabilities in
    - base_model: EMOS model, optional
    """
    fig = plt.figure(figsize=(8, 8))  # Create a figure
    gs = gridspec.GridSpec(4, 1)  # Create a GridSpec with 3 rows and 1 column

    # Create the subplots with different heights
    axs = [plt.subplot(gs[:3]), plt.subplot(gs[3])]

    y_true = y > t
    cdfs = {}

    for name, model in emos_dict.items():
        probs = 1.0 - model.forecast_distribution.comp_cdf(X, t)
        # if probs is greater than 1, set it to 1 or if it is less than 0, set it to 0
        # This is necessary because sometimes the forecast distribution can be slightly off due to numerical errors
        probs = np.clip(probs, 0, 1)
        cdfs[name] = probs.squeeze()

    for name, y_prob in cdfs.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_subset)
        axs[0].plot(prob_pred, prob_true, 'o-', label = name)
    
    if base_model is not None:
        cdf_values = np.clip(1 - base_model.forecast_distribution.comp_cdf(X, t), 0, 1).squeeze()
        prob_true, prob_pred = calibration_curve(y_true, cdf_values, n_bins=n_subset)
        axs[0].plot(prob_pred, prob_true, 'o-', label = "Base model")
        cdfs["Base model"] = cdf_values

    axs[0].plot([0, 1], [0, 1], color="black", linestyle="dashed")
    axs[0].set_xlabel("Mean predicted probability")
    axs[0].set_ylabel("Fraction of positives")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    subset_values = np.linspace(0, 1, n_subset)
    for name, y_prob in cdfs.items():
        counts, bin_edges = np.histogram(y_prob, bins = subset_values)
        counts = np.append(counts, 0)
        axs[1].step(subset_values, counts / len(y) * 100, where='post', label = name)

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Count (%)")
    axs[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
    
def make_reliability_and_sharpness_tf(dict: dict, data: tf.data.Dataset, t: float, n_subset: int = 10, base_model = None, base_model_name = 'Base Model') -> None:
    """
    Makes a reliability diagram and a sharpness diagram for the models in dict, for wind speeds of value t.

    Arguments:
        dict: a dictionary containing the models.
        data: the data for which we make the reliability and sharpness diagram.
        t: the threshold at which we make the diagram.
        n_subset: the number of bins we use to split the data.
        base_model: an optional model that will also be shown in the diagram.
        base_model_name: the name of the base_model.

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
    axs[0].legend()

    subset_values = np.linspace(0, 1, n_subset)
    for name, y_prob in cdfs.items():
        counts, bin_edges = np.histogram(y_prob, bins = subset_values)
        counts = np.append(counts, 0)
        axs[1].step(subset_values, counts / len(y) * 100, where='post', label = name)

    axs[1].set_xlabel("Forecast probability")
    axs[1].set_ylabel("Count (%)")
    axs[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()






