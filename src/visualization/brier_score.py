import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.climatology.climatology import Climatology
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
        brier_scores = model.Brier_Score(data, values)
    elif type(model) == NNForecast:
        brier_scores = np.array(model.Brier_Score(data, values))
    elif isinstance(model, Climatology):
       brier_scores = model.Brier_Score(data, values)
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
    if values[0] < 0.5:
        plt.xlim(0, values[-1])
    else:
        plt.xlim(values[0], values[-1])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    plt.title(title)
    plt.legend()
    plt.show()

def make_bootstrap_sample(X, y):
    # Get the number of samples
    num_samples = X['features_emos'].shape[0]
    
    # Generate bootstrap indices
    indices = np.random.choice(num_samples, num_samples, replace=True)
    
    # Gather all keys in X based on bootstrap indices
    X_bootstrap = {key: tf.gather(value, indices) for key, value in X.items()}
    
    # Gather y based on bootstrap indices
    y_bootstrap = tf.gather(y, indices)
    
    # Create a tf.data.Dataset from the dictionary and y
    dataset = tf.data.Dataset.from_tensor_slices((X_bootstrap, y_bootstrap))
    
    dataset = dataset.batch(len(dataset))
    
    return dataset

def make_bootstrap_bss(basemodel, models, data, values, ylim=None, bootstrap_size=1000):
    # data = data.batch(len(data))
    X, y = next(iter(data))

    base_values = [np.zeros((bootstrap_size, len(values))) for _ in range(len(models))]
    
    if isinstance(basemodel, Climatology):
        base_means = basemodel.Brier_Score(data, values)
        plt.plot([0, 20], [0, 0], 'k--', label='climatology')
        for shift in range(len(models)):
            new_values = values + 0.1 * (shift + 1)
            brier_scores_base_shifted = basemodel.Brier_Score(data, new_values)
            base_values[shift] = brier_scores_base_shifted[:, np.newaxis]
    else:
        base_model_values = np.zeros((bootstrap_size, len(values)))

        for i in range(bootstrap_size):
            dataset = make_bootstrap_sample(X, y)
            base_model_values[i, :] = basemodel.Brier_Score(dataset, values)

            for shift in range(len(models)):
                new_values = values + 0.1 * (shift + 1)
                base_values[shift][i, :] = basemodel.Brier_Score(dataset, new_values)

        base_means = np.mean(base_model_values, axis=0)
        base_means_extra_dim = base_means[np.newaxis, :]
        bss_base = 1 - base_model_values / base_means_extra_dim

        base_bss_mean = np.mean(bss_base, axis=0)
        base_bss_std = np.std(bss_base, axis=0)

        plt.errorbar(values, y=base_bss_mean, yerr=base_bss_std, capsize=2, label='Reference Model')

        for shift in range(len(models)):
            means = np.mean(base_values[shift], axis=0)
            base_values[shift] = means[np.newaxis, :]

    for index, (key, model) in enumerate(models.items()):
        new_values = values + 0.1 * (index + 1)
        brier_scores = np.zeros((bootstrap_size, len(values)))
        is_emos = isinstance(model, EMOS)
        
        for i in range(bootstrap_size):
            dataset = make_bootstrap_sample(X, y)
            if is_emos:
                dataset = dataset.map(lambda X, y: ({'features_emos': X['features_emos']}, y))
            
            brier_scores[i, :] = model.Brier_Score(dataset, values)

        bss_scores = 1 - brier_scores / base_values[index]
        bss_mean = np.mean(bss_scores, axis=0)
        bss_std = np.std(bss_scores, axis=0)

        plt.errorbar(new_values, y=bss_mean, yerr=bss_std, capsize=2, label=key)

    plt.xlim(0, 20)
    plt.xlabel('Threshold')
    plt.ylabel('Brier Skill Scores')
    plt.legend()
    plt.show()










