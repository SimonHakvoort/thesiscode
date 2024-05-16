import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.models.emos import EMOS
from src.models.forecast_distributions import Mixture, MixtureLinear
from src.neural_networks.nn_forecast import NNForecast
from src.visualization.pit import threshold_tf

def plot_forecast_cdf(emos_dict, X, y, observation_value = 0, base_model = None, seed = None, plot_size = 3):
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

    #pick a single random row from y that is greater than the observation value
    candidates = y > observation_value
    indices = np.where(candidates)[0]
    i = np.random.choice(indices, 1)[0]

    min_value = min(y[i] - plot_size, X[i,0] - plot_size)
    max_value = max(y[i] + plot_size, X[i,0] + plot_size)

    x = np.linspace(min_value, max_value, 500)

    #plot the forecast distributions for each model
    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X)

        pdf = distributions.cdf
        y_values = np.zeros((len(x), distributions.batch_shape[0]))
        for p,j in enumerate(x):
            y_values[p,:] = pdf(j).numpy()

        pdf_val_i = y_values[:,i]

        plt.plot(x, pdf_val_i, label = name)

    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X)
        pdf = distributions.cdf
        y_values = np.zeros((len(x), distributions.batch_shape[0]))
        for p,j in enumerate(x):
            y_values[p,:] = pdf(j).numpy()

        pdf_val_i = y_values[:,i]
        plt.plot(x, pdf_val_i, label = 'base model', color = 'black')
    
    #plot the observation
    plt.axvline(y[i], color = 'red', label = 'observation')

    #plot the forecast of the observation
    plt.axvline(X[i,0], color = 'black', label = 'forecast', linestyle = 'dashed')

    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

def plot_forecast_pdf(emos_dict, X, y, observation_value = 0, plot_size = 3, base_model = None, seed = None):
    """
    Plot the forecast distributions for each model in the dictionary, for a single random observation value that is greater than a specified value.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor
    - y: tensor
    - observation_value: float (default 0)
    - plot_size: float (default 3)
    - base_model: EMOS object (optional)
    - seed: int (optional)
    """
    if seed is not None:
        np.random.seed(seed)
    X = X.numpy()
    y = y.numpy()

    #pick a single random row from y that is greater than the observation value
    candidates = y > observation_value
    indices = np.where(candidates)[0]
    i = np.random.choice(indices, 1)[0]

    min_value = min(y[i] - plot_size, X[i,0] - plot_size)
    max_value = max(y[i] + plot_size, X[i,0] + plot_size)

    x = np.linspace(min_value, max_value, 500)

    #plot the forecast distributions for each model
    for name, model in emos_dict.items():
        # use row i and i+1 to get the distribution
        distributions = model.forecast_distribution.get_distribution(X[i-1:i+1, :])

        pdf = distributions.prob
        y_values = np.zeros((len(x), distributions.batch_shape[0]))
        for p,j in enumerate(x):
            y_values[p,:] = pdf(j).numpy()

        pdf_val_i = y_values[:,1]

        plt.plot(x, pdf_val_i, label = name)

    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X[i-1:i+1, :])
        pdf = distributions.prob
        y_values = np.zeros((len(x), distributions.batch_shape[0]))
        for p,j in enumerate(x):
            y_values[p,:] = pdf(j).numpy()

        pdf_val_i = y_values[:,1]
        plt.plot(x, pdf_val_i, label = 'base model', color = 'black')
    
    #plot the observation
    plt.axvline(y[i], color = 'red', label = 'observation')

    #plot the forecast of the observation
    plt.axvline(X[i,0], color = 'black', label = 'forecast', linestyle = 'dashed')

    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

def plot_forecast_pdf_i(emos_dict, X, y, i, plot_size = 3, base_model = None):
    """
    Plot the forecast distributions for each model in the dictionary, for a single observation value.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor
    - y: tensor
    - i: int
    - plot_size: float (default 3)
    - base_model: EMOS object (optional)
    """
    X = X.numpy()
    y = y.numpy()

    min_value = min(y[i] - plot_size, X[i,0] - plot_size)
    max_value = max(y[i] + plot_size, X[i,0] + plot_size)

    x = np.linspace(min_value, max_value, 500)

    #plot the forecast distributions for each model
    for name, model in emos_dict.items():
        distributions = model.forecast_distribution.get_distribution(X)

        pdf = distributions.prob
        y_values = np.zeros((len(x), distributions.batch_shape[0]))
        for p,j in enumerate(x):
            y_values[p,:] = pdf(j).numpy()

        pdf_val_i = y_values[:,i]

        plt.plot(x, pdf_val_i, label = name)

    if base_model is not None:
        distributions = base_model.forecast_distribution.get_distribution(X[i, :])
        pdf = distributions.prob
        plt.plot(x, pdf(x).numpy(), label = 'base model', color = 'black')
    
    #plot the observation
    plt.axvline(y[i], color = 'red', label = 'observation')

    #plot the forecast of the observation
    plt.axvline(X[i,0], color = 'black', label = 'forecast', linestyle = 'dashed')

    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

def plot_weight_mixture(model_dict, values):
    """
    Plot the weight for the distributions for each model as a function of the values.

    Args:
    - model_dict: dictionary of EMOS models
    - values: array

    Returns:
    - None
    """
    for name, model in model_dict.items():
        if type(model.forecast_distribution) == Mixture:
            weight = model.forecast_distribution.get_weight()
            y = weight * np.ones_like(values)
            plt.plot(values, y, label = name)
        if type(model.forecast_distribution) == MixtureLinear:
            weight_a, weight_b = model.forecast_distribution.get_weights()
            # compute f(a + b * x) for x in values and f a sigmoid function
            y = 1 / (1 + np.exp(- (weight_a + weight_b * values)))
            plt.plot(values, y, label = name)

    plt.xlabel('Value')
    plt.ylabel('Weight for first distribution')
    plt.xlim(values[0], values[-1])
    plt.ylim(0, 1)
    plt.legend()
    plt.show()    


def plot_forecast_pdf_tf(model_dict, data, observation_value=0, plot_size=5, base_model=None):
    """
    Plots the probability density function (PDF) of forecasted values for a given model or models.

    Parameters:
    - model_dict (dict): A dictionary containing the models to be plotted. The keys are the names of the models, and the values are the model objects.
    - data (tf.data.Dataset): The input data used for forecasting.
    - observation_value (float): The observed value used as a reference point for plotting.
    - plot_size (float): The range of values to be plotted around the observation value.
    - base_model: The base model to be plotted. If provided, it will be plotted in black color.

    Returns:
    None
    """
    # Function code goes here
    pass
def plot_forecast_pdf_tf(model_dict, data, observation_value = 0, plot_size = 5, base_model = None):

    test_data_greater = threshold_tf(data, observation_value, repeat=False, batching = False)

    test_data_greater = test_data_greater.shuffle(10000)
    

    sample = test_data_greater.take(1)
    X, y = next(iter(sample))
    # add an extra dimension to every tensor in X, add the beginning using
    for key, value in X.items():
        X[key] = tf.expand_dims(value, axis = 0)
    y = tf.expand_dims(y, axis = 0)
        


    for name, model in model_dict.items():
        if isinstance(model, EMOS):
            distributions = model.forecast_distribution.get_distribution(X['features_emos'])
        elif isinstance(model, NNForecast):
            distributions = model.get_distribution(model.predict(X))


        pdf = distributions.prob

        x = np.linspace(y[0] - plot_size, y[0] + plot_size, 500)

        y_values = pdf(x).numpy()

        plt.plot(x, y_values, label = name)

    if base_model is not None:
        if isinstance(base_model, EMOS):
            distributions = base_model.forecast_distribution.get_distribution(X['features_emos'])
        elif isinstance(base_model, NNForecast):
            distributions = base_model.get_distribution(model.predict(X))


        pdf = distributions.prob

        x = np.linspace(y[0] - plot_size, y[0] + plot_size, 500)

        y_values = pdf(x).numpy()

        plt.plot(x, y_values, label = 'base model', color = 'black')

    plt.axvline(y[0], color = 'red', label = 'observation', linestyle = '--')
    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()
            