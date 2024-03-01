import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf

def make_cpit_hist(cdf, y, bins = 20, title = "", t = 0):
    """
    Function to make a PIT histogram for a given cdf and data. The cdf needs to have the same shape as y.
    It is also possible to make a conditional PIT histogram, by setting t to a value different from 0.
    Note that in case t > 0 then we assume that y only contains values greater than 0.

    Args:
    - cdf: cumulative distribution function of shape (n,)
    - y: array with shape (n,) with the true values
    - bins: number of bins for the histogram
    - title: title of the histogram
    - t: real valued number greater than 0

    Returns:
    - None
    """
    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t == 0:
        probabilities = cdf(y)
    else:
        probabilities = (cdf(y) - cdf(t)) / (1 - cdf(t))
    
    plt.hist(probabilities, bins = bins, density = True)
    plt.plot([0, 1], [1, 1], color="black", linestyle="dashed")
    plt.xlabel("Prob")
    plt.ylabel("Obs. freq.")
    plt.title(title)
    plt.show()

def make_cpit_hist_emos(emos, X, y, variance, bins = 20, title = "", t = 0):
    """
    Function to make a PIT histogram for a given EMOS model and data.

    Args:
    - emos: EMOS model
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - bins: number of bins for the histogram
    - title: title of the histogram
    - t: real valued number greater than 0

    Returns:
    - None
    """
    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t > 0:
        X, y, variance = threshold(X, y, variance, t)

    distribution = emos.forecast_distribution.get_distribution(X, variance)
    make_cpit_hist(distribution.cdf, y, bins, title, t)
    
def make_cpit_diagram(cdf_dict, y, title = "", t = 0):
    """
    Function to make a PIT diagram for a given cdf and data. The cdf needs to have the same shape as y, they are stored in cdf_dict
    It is also possible to make a conditional PIT diagram, by setting t to a value different from 0.
    Note that in case t > 0 then we assume that y only contains values greater than 0.

    Args:
    - cdf_dict: dictionary of cdfs with shape (n,)
    - y: array with shape (n,) with the true values
    - title: title of the diagram
    - t: real valued number greater than 0

    Returns:
    - None
    """

    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t == 0:
        for name, cdf in cdf_dict.items():
            probabilities = cdf(y)
            plt.plot(np.sort(probabilities), np.linspace(0, 1, len(probabilities)), label = name)
    else:
        for name, cdf in cdf_dict.items():
            probabilities = (cdf(y) - cdf(t)) / (1 - cdf(t))
            plt.plot(np.sort(probabilities), np.linspace(0, 1, len(probabilities)), label = name)

    plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")

    plt.xlabel("Prob")
    plt.ylabel("Obs. freq.")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def make_cpit_diagram_emos(emos_dict, X, y, variance, title = "", t = 0):
    """
    Function to make a PIT diagram for a dictionary of EMOS models and data.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - title: title of the diagram
    - t: real valued number greater than 0

    Returns:
    - None
    """
    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t > 0:
        X, y, variance = threshold(X, y, variance, t)

    cdf_dict = {}
    for name, emos in emos_dict.items():
        distribution = emos.forecast_distribution.get_distribution(X, variance)
        cdf_dict[name] = distribution.cdf

    make_cpit_diagram(cdf_dict, y, title, t)


def threshold(X, y, variance, t):
    """
    Checks for which indices in y the value is greater than t, and then returns the rows of X, y, variance for which this is the case

    Args:
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values

    Returns:
    - X: tensor
    - y: tensor
    - variance: tensor
    """
    indices = tf.where(y > t)

    y_greater = tf.gather(y, indices)
    X_greater = tf.gather(X, indices)
    variance_greater = tf.gather(variance, indices)
    return X_greater, y_greater, variance_greater

    


