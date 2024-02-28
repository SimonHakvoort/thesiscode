import matplotlib.pyplot as plt
import numpy as np

def make_pit_hist(emos, X, y, variance, bins = 20, title = "PIT histogram"):
    """
    Function to make a PIT histogram for a given EMOS model and data.

    Args:
    - emos: EMOS model
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - bins: number of bins for the histogram
    - title: title of the histogram

    Returns:
    - None

    """
    distribution = emos.forecast_distribution(X, variance)
    probabilities = distribution.cdf(y)

    plt.hist(probabilities, bins=bins, density=True)
    plt.title(title)

    plt.show()

def make_pit_diagram(emos, X, y, variance, title = "PIT diagram"):
    """
    Function to make a PIT diagram for a given EMOS model and data.

    Args:
    - emos: EMOS model
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - title: title of the diagram

    Returns:
    - None

    """
    distribution = emos.forecast_distribution(X, variance)
    probabilities = distribution.cdf(y)

    plt.scatter(np.sort(probabilities), np.linspace(0, 1, len(probabilities)))
    plt.title(title)

    plt.show()




