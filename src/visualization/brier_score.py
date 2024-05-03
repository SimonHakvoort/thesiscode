import matplotlib.pyplot as plt
import numpy as np

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
        for i, threshold in enumerate(values):
            brier_scores[i] = model.Brier_Score_tfdataset(data, threshold).numpy()
    elif type(model) == NNForecast:
        brier_scores = np.array(model.Brier_Score(data, values))
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

def make_brier_skill_plot_tf(basemodel, models, data, values, ylim = None, title = 'Brier skill score'):
    brier_base_model = get_brier_scores_tf(basemodel, data, values)
    for model in models:
        brier_scores = get_brier_scores_tf(models[model], data, values)
        brier_skill_scores = 1 - brier_scores / brier_base_model
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