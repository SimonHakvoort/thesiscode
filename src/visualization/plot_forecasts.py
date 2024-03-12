import matplotlib.pyplot as plt
import numpy as np

def plot_forecast_cdf(emos_dict, X, y, variances, observation_value = 0, base_model = None, seed = None):
    """
    Plot the forecast distributions for each model in the dictionary, for a single random observation value that is greater than a specified value.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor
    - y: tensor
    - variances: tensor
    - observation_value: float (default 0)
    - base_model: EMOS object (optional)
    - seed: int (optional)
    """
    if seed is not None:
        np.random.seed(seed)
    y = y.numpy()
    X = X.numpy()
    variances = variances.numpy()

    #pick a single random row from y that is greater than the observation value
    candidates = y > observation_value
    indices = np.where(candidates)[0]
    i = np.random.choice(indices, 1)[0]
    x = np.linspace(y[i] - 3, y[i] + 3, 100)

    #plot the forecast distributions for each model
    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X[i, :], variances[i])
        cdf = distributions.cdf

        plt.plot(x, cdf(x).numpy(), label = name)
    
    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X[i, :], variances[i])
        cdf = distributions.cdf
        plt.plot(x, cdf(x).numpy(), label = 'base model', color = 'black', linestyle = 'dashed')
    
    #plot the observation
    plt.axvline(y[i], color = 'red', label = 'observation')

    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

def plot_forecast_pdf(emos_dict, X, y, variances, observation_value = 0, base_model = None, seed = None):
    """
    Plot the forecast distributions for each model in the dictionary, for a single random observation value that is greater than a specified value.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor
    - y: tensor
    - variances: tensor
    - observation_value: float (default 0)
    - base_model: EMOS object (optional)
    - seed: int (optional)
    """
    if seed is not None:
        np.random.seed(seed)
    X = X.numpy()
    y = y.numpy()
    variances = variances.numpy()

    #pick a single random row from y that is greater than the observation value
    candidates = y > observation_value
    indices = np.where(candidates)[0]
    i = np.random.choice(indices, 1)[0]
    x = np.linspace(y[i] - 3, y[i] + 3, 100)

    #plot the forecast distributions for each model
    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X[i, :], variances[i])
        pdf = distributions.prob

        plt.plot(x, pdf(x).numpy(), label = name)

    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X[i, :], variances[i])
        pdf = distributions.prob
        plt.plot(x, pdf(x).numpy(), label = 'base model', color = 'black', linestyle = 'dashed')
    
    #plot the observation
    plt.axvline(y[i], color = 'red', label = 'observation')

    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()
