import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.linreg_emos.emos import LinearEMOS
from src.linreg_emos.forecast_distributions import Mixture, MixtureLinear
from src.cnn_emos.nn_distributions import NNMixture
from src.cnn_emos.nn_forecast import CNNEMOS
from src.visualization.pit import threshold_tf


def plot_weight_mixture(model_dict: dict, values: np.ndarray) -> None:
    """
    Plot the weight for the distributions for each model as a function of the values.
    model_dict should contain EMOS models as value, where the forecast distribution is 
    Mixture of MixtureLienar.

    Args:
        model_dict: dictionary of EMOS models.
        values: array for which we want to find the weight.

    Returns:
        None
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


def plot_forecast_pdf_tf(model_dict, data, observation_value = 0, plot_size = 5, base_model = None):
    """
    Plots the probability density function (PDF) of forecasted values for a given model or models.

    Parameters:
        model_dict (dict): A dictionary containing the models to be plotted. The keys are the names of the models, and the values are the model objects.
        data (tf.data.Dataset): The input data used for forecasting.
        observation_value (float): The observed value used as a reference point for plotting.
        plot_size (float): The range of values to be plotted around the observation value.
        base_model: The base model to be plotted. If provided, it will be plotted in black color.

    Returns:
        None
    """
    test_data_greater = threshold_tf(data, observation_value, repeat=False, batching=False)

    test_data_greater = test_data_greater.shuffle(10000)
    

    sample = test_data_greater.take(1)
    X, y = next(iter(sample))

    forecast = X['wind_speed_forecast']
    # add an extra dimension to every tensor in X, add the beginning using
    for key, value in X.items():
        X[key] = tf.expand_dims(value, axis = 0)
    y = tf.expand_dims(y, axis = 0)
        
    

    for name, model in model_dict.items():
        if isinstance(model, LinearEMOS):
            distributions = model.forecast_distribution.get_distribution(X['features_emos'])
        elif isinstance(model, CNNEMOS):
            distributions = model.get_distribution(model.predict(X))


        pdf = distributions.prob

        x = np.linspace(y[0] - plot_size, y[0] + plot_size, 500)

        y_values = pdf(x).numpy()

        plt.plot(x, y_values, label = name)

    if base_model is not None:
        if isinstance(base_model, LinearEMOS):
            distributions = base_model.forecast_distribution.get_distribution(X['features_emos'])
        elif isinstance(base_model, CNNEMOS):
            distributions = base_model.get_distribution(model.predict(X))


        pdf = distributions.prob

        x = np.linspace(y[0] - plot_size, y[0] + plot_size, 500)

        y_values = pdf(x).numpy()

        plt.plot(x, y_values, label = 'base model', color = 'black')

    plt.axvline(forecast, color = 'black', label = 'forecast', linestyle = 'dashed')
    plt.axvline(y[0], color = 'red', label = 'observation', linestyle = '--')
    plt.xlabel('Value')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()

def plot_weight_mixture_cnns(model_dict: dict, data: tf.data.Dataset) -> None:
    """
    Plots the weight of the mixture distribution for EMOS with CNNs.
    The forecast distribution should be NNMixture, where the first distribution is TN and the 
    second distribution is the LN.

    Arguments:
        model_dict (dict): a dictionary containing NNForecast objects. 
        data (tf.data.Dataset): the dataset for which we plot the weights.
    """
    X, y = next(iter(data))

    for name, model in model_dict.items():
        if not isinstance(model.model.get_forecast_distribution(), NNMixture):
            raise ValueError("We can only plot the weights in case the forecast distribution is NNMixture!")
        
        y_pred = model.predict(X)
        weight = y_pred[:, 0]

        plt.scatter(y, weight, alpha=0.5, label=name, marker='.')

    plt.xlabel("Observed Wind Speed")
    plt.ylabel("Weight for TN")
    plt.ylim(0,1)
    plt.xlim(0, np.max(y))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()
    plt.show()

            