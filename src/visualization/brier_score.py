import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
from src.models.emos import EMOS
from src.neural_networks.nn_forecast import NNForecast


def make_brier_plot(emos, X, y, values, title = ''):
    """
    Makes a plot of the Brier score for the model. We assume that X is already normalized.

    Args:
    - emos: EMOS object
    - X: tensor
    - y: tensor
    - values: list of floats on which to compute the Brier score
    - title: string

    Returns:
    - None
    """
    brier_scores = get_brier_scores(emos, X, y, values)

    plt.plot(values, brier_scores)
    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier score')
    plt.title(title)
    plt.xlim(values[0], values[-1])
    #ensure that ylim has a minimum of 0
    plt.ylim(0, max(brier_scores))
    plt.show()

def get_brier_scores(emos, X, y, values):
    """
    Returns the Brier scores for a range of thresholds. We assume that X is already normalized.

    Args:
    - emos: EMOS object
    - X: tensor
    - y: tensor
    - values: list of floats on which to compute the Brier score

    Returns:
    - brier_scores: list of floats
    """
    brier_scores = np.zeros(len(values))
    for i, threshold in enumerate(values):
        brier_scores[i] = emos.Brier_Score(X, y, threshold)
    return brier_scores

def get_brier_scores_tf(model, data, values):
    brier_scores = np.zeros(len(values))
    if type(model) == EMOS:
        brier_scores = model.Brier_Score(data, values)
    elif type(model) == NNForecast:
        brier_scores = np.array(model.Brier_Score(data, values))
    elif isinstance(model, Climatology):
       brier_scores = model.Brier_Score(data, values)
    else:
        raise ValueError('Model type not recognized')
    return brier_scores


def get_brier_skill_scores(emos1, emos2, X, y, values):
    """
    Returns the Brier skill scores for a range of thresholds. We assume that X is already normalized.

    Args:
    - emos1: EMOS object
    - emos2: EMOS object
    - X: tensor
    - y: tensor
    - values: list of floats on which to compute the Brier score

    Returns:
    - brier_skill_scores: list of floats
    """
    brier_scores1 = get_brier_scores(emos1, X, y, values)
    brier_scores2 = get_brier_scores(emos2, X, y, values)
    return 1 - brier_scores1 / brier_scores2

def make_brier_skill_plot(basemodel, models, X, y, values, ylim = None, title = 'Brier skill score'):
    """
    Makes a plot of the Brier skill score for the models. We assume that X is already normalized. basemodel is the model to compare to. 
    Includes a legend with the names of the models.

    Args:
    - basemodel: EMOS object
    - models: dictionary of EMOS objects
    - X: tensor
    - y: tensor
    - values: list of floats on which to compute the Brier score
    - ylim: tuple of floats
    - title: string
    """
    for model in models:
        brier_skill_scores = get_brier_skill_scores(models[model], basemodel, X, y, values)
        plt.plot(values, brier_skill_scores, label = model)

    # print a striped black horizontal line at y=0
    plt.axhline(0, color='black', linestyle='--')

    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier skill score')
    plt.xlim(values[0], values[-1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title)
    plt.legend()
    plt.show()

def make_brier_skill_plot_tf(basemodel, models, data, values, ylim = None, title = None, name_base_model = 'Reference Model'):
    """
    Plots the Brier skill score (BSS) for the models, which is a dict for the numbers in values. 
    Evaluates the performance for a single batch in data.

    Arguments:
        basemodel: reference model
        models (dict): models to compare to basemodel
        data (tf.data.Dataset): data to compute Brier scores.
        values (np.array): values to compute the BSS.
        ylim (tuple, optional): tuple specifying the range of the y-axis.
        title (str, optional: Title for the plot.
        name_base_model (str, optional): name for the reference model in the legend.
    """
    # brier_base_model = get_brier_scores_tf(basemodel, data, values)
    brier_base_model = basemodel.Brier_Score(data, values)
    for model in models:
        # brier_scores = get_brier_scores_tf(models[model], data, values)
        brier_scores = models[model].Brier_Score(data, values)
        brier_skill_scores = 1 - brier_scores / brier_base_model
        plt.plot(values, brier_skill_scores, label = model)

    # print a striped black horizontal line at y=0
    plt.axhline(0, color='black', linestyle='--')

    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier skill score')
    if values[0] < 0.5:
        plt.xlim(0, values[-1])
    else:
        plt.xlim(values[0], values[-1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    
    if title is not None:
        plt.title(title)

    plt.legend()
    plt.show()

def make_bootstrap_sample(X: dict, y: tf.Tensor) -> tf.data.Dataset:
    # Get the number of samples
    num_samples = X['features_emos'].shape[0]
    
    indices = np.random.choice(num_samples, num_samples, replace=True)
    
    X_bootstrap = {key: tf.gather(value, indices) for key, value in X.items()}
    
    y_bootstrap = tf.gather(y, indices)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_bootstrap, y_bootstrap))
    
    dataset = dataset.batch(len(dataset))
    
    return dataset

def make_bootstrap_brier(basemodel, models, data, values, ylim=None, bootstrap_size=1000, title = None, name_base_model = None):
    scores = {key: np.zeros((bootstrap_size, len(values))) for key in models.keys()}

    base_brier_scores = np.zeros((bootstrap_size, len(values)))

    X, y = next(iter(data))

    for i in range(bootstrap_size):
        bootstrap_data = make_bootstrap_sample(X, y)

        base_brier_scores[i, :] = basemodel.Brier_Score(bootstrap_data, values)

        for j, name in enumerate(models.keys()):
            new_values = values + (j + 1) * 0.1

            base_scores_shift = basemodel.Brier_Score(bootstrap_data, new_values)

            model_scores = models[name].Brier_Score(bootstrap_data, new_values)

            bss = 1 - model_scores / base_scores_shift

            scores[name][i, :] = bss
        
    for i, name in enumerate(models.keys()):
        score = scores[name]

        new_values = values + (i + 1) * 0.1

        mean = np.mean(score, axis=0)
        std = np.std(score, axis=0)

        plt.errorbar(new_values, y=mean, yerr=std, capsize=2, label=name)

    base_means = np.mean(base_brier_scores, axis=0)
    bss_scores_base = 1 - base_brier_scores / base_means[np.newaxis, :]
    base_bss_mean = np.mean(bss_scores_base, axis=0)
    base_bss_std = np.std(bss_scores_base, axis=0)

    # plt.errorbar(values, y=base_bss_mean, yerr=base_bss_std, capsize=2, label='Base Model')
    if isinstance(basemodel, Climatology):
        plt.plot(values, base_bss_mean, label='Climatology')
    else:
        if name_base_model is None:
            naming = 'Reference Model'
        else:
            naming = name_base_model
        plt.plot(values, base_bss_mean, label=naming)
    
    plt.xlim(0, 20)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.xlabel('Threshold')
    plt.ylabel('Brier Skill Scores')
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()









