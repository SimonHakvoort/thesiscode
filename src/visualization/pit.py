import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
from src.linreg_emos.emos import BaseForecastModel, LinearEMOS
from src.cnn_emos.nn_forecast import CNNEMOS

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
    

def make_cpit_diagram(cdf_dict: dict[str, BaseForecastModel], y: np.ndarray, title: str = "", t: float = 0.0, gev_shape = None):
    """
    Function to make a PIT diagram for a given cdf and data. The cdf needs to have the same shape as y, they are stored in cdf_dict
    It is also possible to make a conditional PIT diagram, by setting t to a value different from 0.
    Note that in case t > 0 then we assume that y only contains values greater than 0.

    Args:
        cdf_dict: dictionary of cdfs with shape (n,)
        y: array with shape (n,) with the true values
        title: title of the diagram
        t: real valued number greater than 0

    Returns:
        None
    """

    if t < 0:
        raise ValueError("t needs to be greater than 0")
    elif t == 0:
        for name, cdf in cdf_dict.items():
            probabilities = cdf(y)
            if gev_shape[name] is not None:
                probabilities = np.array(probabilities)  
                shape_values = np.array(gev_shape[name])
                
                # change the probabilities such that nan at i becomes 1 if gev_shape[i] < 0 and 0 if gev_shape[i] > 0
                nan_indices = np.isnan(probabilities)
        
                # Replace NaN values based on the corresponding gev_shape values
                probabilities[nan_indices & (shape_values < 0)] = 1
                probabilities[nan_indices & (shape_values > 0)] = 0

            plt.plot(np.sort(probabilities), np.linspace(0, 1, len(probabilities)), label = name)
            
    else:
        for name, cdf in cdf_dict.items():
            probabilities = (cdf(y) - cdf(t)) / (1 - cdf(t))
            if gev_shape[name] is not None:
                probabilities = np.array(probabilities)  
                shape_values = np.array(gev_shape[name])

                # change the probabilities such that nan at i becomes 1 if gev_shape[i] < 0 and 0 if gev_shape[i] > 0
                nan_indices = np.isnan(probabilities)
            
                # Replace NaN values based on the corresponding gev_shape values
                probabilities[nan_indices & (shape_values < 0)] = 1
                probabilities[nan_indices & (shape_values > 0)] = 0  


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
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
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

    dataset_length = [i for i,_ in enumerate(filtered_data)][-1] + 1

    if batching:
        filtered_data = filtered_data.batch(dataset_length)
    if repeat:
        filtered_data = filtered_data.repeat()

    return filtered_data

def make_cpit_diagram_tf(model_dict: dict[str, BaseForecastModel], data: tf.data.Dataset, t: int = 0, base_model: BaseForecastModel = None, base_model_name: str = 'Base Model'):
    """
    A a conditional PIT diagram for all the models in model_dict. 

    Arguments:
        model_dict (dict[str, BaseForecastModel]): dictionary of models.
        data (tf.data.Dataset): data for which we compute the cPIT (unfiltered and unbatched).
        t (int): threshold.
        base_model (BaseForecastModel, optional): in case you want to compare it for an extra model.
        base_model_name (optional): naming of the base_model for the legend.
    """
    # We first compute the filtered data based on the threshold.
    data_filtered = threshold_tf(data, t)
    X, y = next(iter(data_filtered)) 
    cdf_dict = {}
    gev_shapes = {}
    for name, model in model_dict.items():
        if isinstance(model, Climatology):
            raise ValueError("Climatology has not been implemented for cpit diagrams!")
        distribution, observations = model.get_prob_distribution(data_filtered)
        cdf_dict[name] = distribution.cdf
        gev_shapes[name] = model.get_gev_shape(X)

        
    if base_model is not None:
        if isinstance(model, Climatology):
            raise ValueError("Climatology has not been implemented for cpit diagrams!")
        distribution, observations = base_model.get_prob_distribution(data_filtered)
        cdf_dict[base_model_name] = distribution.cdf
        gev_shapes[base_model_name] = model.get_gev_shape(X)
        
    make_cpit_diagram(cdf_dict, observations, t=t, gev_shape=gev_shapes)

def comp_pit_score_tf(model, data, t = 0):
    # this function integrates the area between the pit curve and the diagonal
    # the closer the score is to 0, the better the model
    test_data_greater = threshold_tf(data, t)

    if isinstance(model, LinearEMOS):
        distribution, observations = model.get_prob_distribution(test_data_greater)
        cdf = distribution.cdf
    elif isinstance(model, CNNEMOS):
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
