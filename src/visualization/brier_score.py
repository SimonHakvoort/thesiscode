import matplotlib.pyplot as plt
import numpy as np


def brier_plot(emos, X, y, variances, values, title = ''):
    """
    Makes a plot of the Brier score for the model. We assume that X is already normalized.

    Args:
    - emos: EMOS object
    - X: tensor
    - y: tensor
    - variances: tensor
    - values: list of floats on which to compute the Brier score
    - title: string

    Returns:
    - None
    """
    brier_scores = get_brier_scores(emos, X, y, variances, values)

    plt.plot(values, brier_scores)
    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier score')
    plt.title(title)
    plt.show()

def get_brier_scores(emos, X, y, variances, values):
    """
    Returns the Brier scores for a range of thresholds. We assume that X is already normalized.

    Args:
    - emos: EMOS object
    - X: tensor
    - y: tensor
    - variances: tensor
    - values: list of floats on which to compute the Brier score

    Returns:
    - brier_scores: list of floats
    """
    brier_scores = np.zeros(len(values))
    for i, threshold in enumerate(values):
        brier_scores[i] = emos.loss_Brier_score(X, y, variances, threshold)
    return brier_scores

def get_brier_skill_scores(emos1, emos2, X, y, variances, values):
    """
    Returns the Brier skill scores for a range of thresholds. We assume that X is already normalized.

    Args:
    - emos1: EMOS object
    - emos2: EMOS object
    - X: tensor
    - y: tensor
    - variances: tensor
    - values: list of floats on which to compute the Brier score

    Returns:
    - brier_skill_scores: list of floats
    """
    brier_scores1 = get_brier_scores(emos1, X, y, variances, values)
    brier_scores2 = get_brier_scores(emos2, X, y, variances, values)
    return 1 - brier_scores1 / brier_scores2

def brier_skill_plot(basemodel, models, X, y, variances, values, title = 'Brier skill score'):
    """
    Makes a plot of the Brier skill score for the models. We assume that X is already normalized. basemodel is the model to compare to. 
    Includes a legend with the names of the models.

    Args:
    - basemodel: EMOS object
    - models: dictionary of EMOS objects
    - X: tensor
    - y: tensor
    - variances: tensor
    - values: list of floats on which to compute the Brier score
    """
    for model in models:
        brier_skill_scores = get_brier_skill_scores(models[model], basemodel, X, y, variances, values)
        plt.plot(values, brier_skill_scores, label = model)
    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier skill score')
    plt.xlim(values[0], values[-1])
    plt.legend()
    plt.show()