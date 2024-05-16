import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf

from src.models.emos import EMOS
from src.neural_networks.nn_forecast import NNForecast

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

    print("There are", len(probabilities), "values in the PIT histogram")
    
    plt.hist(probabilities, bins = bins, density = True, rwidth=0.9)
    plt.plot([0, 1], [1, 1], color="black", linestyle="dashed")
    plt.xlabel("Prob")
    plt.ylabel("Obs. freq.")
    plt.title(title)
    plt.show()

def make_cpit_hist_emos(emos, X, y, bins = 20, title = "", t = 0):
    """
    Function to make a PIT histogram for a given EMOS model and data.

    Args:
    - emos: EMOS model
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
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

    distribution = emos.forecast_distribution.get_distribution(X)
    make_cpit_hist(distribution.cdf, y, bins, title, t)
    

def make_cpit_diagram(cdf_dict, y, title = "", t = 0.0, gev_shape = None):
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
            if gev_shape is not None:
                # change the probabilities such that nan becomes 1 if gev_shape < 0 and 0 if gev_shape > 0
                shape = gev_shape[name]
                if shape is not None:
                    probabilities = np.where(np.isnan(probabilities), 1 if shape < 0 else 0, probabilities)
            
            # print contain nan if probabilities contains nan
            if np.isnan(probabilities).any():
                print("probabilities contain nan")

            plt.plot(np.sort(probabilities), np.linspace(0, 1, len(probabilities)), label = name)
            
    else:
        for name, cdf in cdf_dict.items():
            probabilities = (cdf(y) - cdf(t)) / (1 - cdf(t))
            if gev_shape is not None:
                # change the probabilities such that nan becomes 1 if gev_shape < 0 and 0 if gev_shape > 0
                shape = gev_shape[name]
                if shape is not None:
                    probabilities = np.where(np.isnan(probabilities), 1 if shape < 0 else 0, probabilities)

            #remove the nan from probabilities
            # these can occur if cdf(t) = 1, which occurs in the dataset for large t.
            # therefore there is often one value removed from the diagram
            probabilities = tf.boolean_mask(probabilities, tf.math.is_finite(probabilities))
            probabilities = tf.sort(probabilities)

            plt.plot(np.sort(probabilities), np.linspace(0, 1, len(probabilities)), label = name)

    print("There are", len(probabilities), "values in the PIT diagram")

    plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")

    plt.xlabel("Prob")
    plt.ylabel("Obs. freq.")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def make_cpit_diagram_emos(emos_dict, X, y, title = "", t = 0, base_model = None):
    """
    Function to make a PIT diagram for a dictionary of EMOS models and data.

    Args:
    - emos_dict: dictionary of EMOS models
    - X: tensor with shape (n, m), where n is the number of samples and m is the number of features
    - y: array with shape (n,) with the true values
    - variances: array with shape (n,) with the variances in the neighborhood of the true values
    - title: title of the diagram
    - t: real valued number greater than 0
    - base_model: EMOS model, optional

    Returns:
    - None
    """
    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t > 0:
        X, y = threshold(X, y, t)

    cdf_dict = {}
    for name, emos in emos_dict.items():
        distribution = emos.forecast_distribution.get_distribution(X)
        cdf_dict[name] = distribution.cdf

    gev_shape = {}

    if base_model is not None:
        distribution = base_model.forecast_distribution.get_distribution(X)
        cdf_dict["Base Model"] = distribution.cdf
        gev_shape["Base Model"] = base_model.forecast_distribution.get_gev_shape()


    for name, emos in emos_dict.items():
        gev_shape[name] = emos.forecast_distribution.get_gev_shape()

    make_cpit_diagram(cdf_dict, y, title, t, gev_shape)




def threshold(X, y, t):
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
    indices = tf.reshape(indices, [-1])

    y_greater = tf.gather(y, indices)
    X_greater = tf.gather(X, indices)
    #variance_greater = tf.gather(variance, indices)
    return X_greater, y_greater

def threshold_tf(data, t, repeat = True, batching = True):
    def filter_function(X, y):
        return y > t
        
    filtered_data = data.filter(filter_function)

    if batching:
        filtered_data = filtered_data.batch(100000)
    if repeat:
        filtered_data = filtered_data.repeat()
    #X, y = next(iter(filtered_data))

    return filtered_data

def make_cpit_diagram_tf(model_dict, data, t = 0, base_model = None):
    data = threshold_tf(data, t)
    
    cdf_dict = {}
    for name, model in model_dict.items():
        if type(model) == EMOS:
            distribution, observations = model.get_prob_distribution(data)
            cdf_dict[name] = distribution.cdf
        elif type(model) == NNForecast:
            distribution, observations = model.get_prob_distribution(data)
            cdf_dict[name] = distribution.cdf
        else:
            raise ValueError('Model type not recognized')
        
    if base_model is not None:
        if type(model) == EMOS:
            distribution, observations = base_model.get_prob_distribution(data)
            cdf_dict['base_model'] = distribution.cdf
        elif type(model) == NNForecast:
            distribution, observations = base_model.get_prob_distribution(data)
            cdf_dict['base_model'] = distribution.cdf
        else:
            raise ValueError('Model type not recognized')
        
    make_cpit_diagram(cdf_dict, observations, t=t)

def comp_pit_score_tf(model, data, t = 0):
    # this function integrates the area between the pit curve and the diagonal
    # the closer the score is to 0, the better the model
    test_data_greater = threshold_tf(data, t)

    if isinstance(model, EMOS):
        distribution, observations = model.get_prob_distribution(test_data_greater)
        cdf = distribution.cdf
    elif isinstance(model, NNForecast):
        distribution, observations = model.get_prob_distribution(test_data_greater)
        cdf = distribution.cdf
    else:
        raise ValueError('Model type not recognized')
    
    if t == 0:
        probabilities = cdf(observations)
    elif t > 0:
        #probabilities = (cdf(observations) - cdf(t)) / (1 - cdf(t))
        upper = cdf(observations) - cdf(t)
        lower = 1 - cdf(t)
        # remove the points where lower is 0
        mask = tf.where(lower == 0, False, True)
        upper = tf.boolean_mask(upper, mask)
        lower = tf.boolean_mask(lower, mask)
        probabilities = upper / lower
    else:
        raise ValueError('t needs to be greater than 0')
    
    probabilities = tf.sort(probabilities)

    return np.mean(np.abs(probabilities - np.linspace(0, 1, len(probabilities))))

def comp_multiple_pit_scores(model_dict, data, t = 0, base_model = None):
    scores = {}
    for name, model in model_dict.items():
        scores[name] = comp_pit_score_tf(model, data, t)
    
    if base_model is not None:
        scores['base_model'] = comp_pit_score_tf(base_model, data, t)
    
    return scores
