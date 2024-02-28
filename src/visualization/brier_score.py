import matplotlib.pyplot as plt
import numpy as np




def brier_plot(emos, X, y, variances, title = 'Brier score'):
    """
    Makes a plot of the Brier score for the model. We assume that X is already normalized.

    Args:
    - emos: EMOS object
    - X: tensor
    - y: tensor
    - variances: tensor
    - title: string

    Returns:
    - None
    """
    thresholds = np.linspace(0, 20, 2000)
    brier_scores = np.zeros(thresholds.shape)
    for i, threshold in enumerate(thresholds):
        brier_scores[i] = emos.brier_score(X, y, variances, threshold)

    plt.plot(thresholds, brier_scores)
    plt.xlabel('wind speed threshold (m/s)')
    plt.ylabel('Brier score')
    plt.title(title)
    plt.show()
