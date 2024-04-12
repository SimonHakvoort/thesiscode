import pickle as pkl

import numpy as np

from src.models.emos import EMOS
from src.models.forecast_distributions import Mixture, TruncatedGEV, TruncatedNormal, LogNormal, GEV, Frechet, DistributionMixture, MixtureLinear, distribution_name
from src.models.get_data import get_normalized_tensor


def train_model(forecast_distribution, loss, optimizer, learning_rate, folds, all_features, location_features, scale_features, neighbourhood_size, ignore, epochs, **kwargs):
    setup = {}

    setup['forecast_distribution'] = distribution_name(forecast_distribution)
    setup['loss'] = loss
    setup['optimizer'] = optimizer
    setup['learning_rate'] = learning_rate

    if 'samples' in kwargs:
        setup['samples'] = kwargs['samples']
    
    pretrained = False

    if setup['forecast_distribution'] == 'distr_mixture' or setup['forecast_distribution'] == 'distr_mixture_linear':
        setup['distribution_1'] = distribution_name(kwargs['distribution_1'])
        setup['distribution_2'] = distribution_name(kwargs['distribution_2'])
        if 'pretrained' in kwargs:
            pretrained = kwargs['pretrained']
            original_distribution = setup['forecast_distribution']

    data = get_normalized_tensor(neighbourhood_size, all_features, folds, ignore)
    X = data['X']
    y = data['y']
    setup['feature_mean'] = data['mean']
    setup['feature_std'] = data['std']

    setup['folds'] = folds

    setup['all_features'] = all_features
    setup['location_features'] = location_features
    setup['scale_features'] = scale_features
    setup["neighbourhood_size"] = neighbourhood_size

    if 'subset_size' in kwargs:
        subset_size = kwargs['subset_size']
    else:
        subset_size = None

    if 'chain_function' in kwargs:
        setup['chain_function'] = kwargs['chain_function']

        if 'chain_function_mean' in kwargs:
            setup['chain_function_mean'] = kwargs['chain_function_mean']
        if 'chain_function_std' in kwargs:
            setup['chain_function_std'] = kwargs['chain_function_std']
        if 'chain_function_threshold' in kwargs:
            setup['chain_function_threshold'] = kwargs['chain_function_threshold']
        if 'chain_function_constant' in kwargs:
            setup['chain_function_constant'] = kwargs['chain_function_constant']
    
    if "random_init" in kwargs:
        setup['random_init'] = kwargs['random_init']

    if pretrained:
        setup['forecast_distribution'] = setup['distribution_1']
        model_1 = EMOS(setup)

        setup['forecast_distribution'] = setup['distribution_2']
        model_2 = EMOS(setup)

        model_1.fit(X, y, 100, False, subset_size = subset_size)
        model_2.fit(X, y, 100, False, subset_size = subset_size)
        setup['parameters'] = {**model_1.get_parameters(), **model_2.get_parameters()}
        setup['forecast_distribution'] = original_distribution

    model = EMOS(setup)

    
    if 'parameters' in kwargs:
        model.set_parameters(kwargs['parameters'])
    
    printing = False
    if 'printing' in kwargs:
        printing = kwargs['printing']



    model.fit(X, y, epochs, printing = printing, subset_size = subset_size)

    return model



    
def save_model(model, path = '/net/pc200239/nobackup/users/hakvoort/models/emos/'):
    folder = model.forecast_distribution.folder_name() + '/'

    distribution_name = model.forecast_distribution.distribution_name()
              

    if model.loss.__name__ == 'loss_CRPS_sample':
        loss = 'crps'

    chain_specs = ''

    if model.loss.__name__ == 'loss_twCRPS_sample':
        loss = 'twcrps'
        
        if model.chain_function.__name__ == 'chain_function_normal_cdf':
            mean = model.chain_function_mean.numpy()
            std = model.chain_function_std.numpy()
            chain_specs = f'mean{mean}_std{std}'

        if model.chain_function.__name__ == 'chain_function_indicator':
            threshold = model.chain_function_threshold.numpy()
            chain_specs = f'threshold{threshold}'

        if model.chain_function.__name__ == 'chain_function_normal_cdf_plus_constant':
            mean = model.chain_function_mean.numpy()
            std = model.chain_function_std.numpy()
            constant = model.chain_function_constant.numpy()
            # round the constant to 5 decimals
            constant = np.around(constant, 3)
            chain_specs = f'mean{mean}_std{std}_constant{constant}'

    if model.loss.__name__ == 'loss_log_likelihood':
        loss = 'loglik'

    epochs = model.steps_made
    name = f'{distribution_name}_{loss}_{chain_specs}_epochs{epochs}'

    folds = model.folds
    fold_info = "_folds"
    for fold in folds:
        fold_info += f'_{fold}'
    
    name += fold_info

    # a list containing indices
    mean_features = model.forecast_distribution.location_features_indices
    std_features = model.forecast_distribution.scale_features_indices

    mean_features_str = ''
    for feature in mean_features:
        mean_features_str += f'{feature}_'

    std_features_str = ''
    for feature in std_features:
        std_features_str += f'{feature}_'

    name += f'_mean{mean_features_str}std{std_features_str}'

    model_dict = model.to_dict()
    with open(f'{path}{folder}{name}.pkl', 'wb') as f:
        pkl.dump(model_dict, f)

    print(f'Model saved as {path}{folder}{name}.pkl')

    
def train_and_save(forecast_distribution, loss, optimizer, learning_rate, folds, all_features, location_features, scale_features, neighbourhood_size, ignore, epochs, **kwargs):
    model = train_model(forecast_distribution, loss, optimizer, learning_rate, folds, all_features, location_features, scale_features, neighbourhood_size, ignore, epochs, **kwargs)
    save_model(model)
    return model

def train_and_save_multiple_models_twCRPS(forecast_distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, thresholds, mean_and_stds, include_mixture, include_mixture_linear, **kwargs):
    for distr in forecast_distribution:
        for mean_std in mean_and_stds:
            chain_function = 'chain_function_normal_cdf'
            chain_function_mean = mean_std[0]
            chain_function_std = mean_std[1]
            model = train_model(distr, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_mean = chain_function_mean, chain_function_std = chain_function_std, **kwargs)
            save_model(model)
        
        for threshold in thresholds:
            chain_function = 'chain_function_indicator'
            model = train_model(distr, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_threshold = threshold, **kwargs)
            save_model(model)


    if include_mixture:
        for distr in forecast_distribution:
            distribution_1 = 'distr_trunc_normal'
            distribution_2 = distr
            distribution = 'distr_mixture'
            for mean_std in mean_and_stds:
                chain_function = 'chain_function_normal_cdf'
                chain_function_mean = mean_std[0]
                chain_function_std = mean_std[1]
                model = train_model(distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_mean = chain_function_mean, chain_function_std = chain_function_std, distribution_1 = distribution_1, distribution_2 = distribution_2, **kwargs)
                save_model(model)
            
            for threshold in thresholds:
                chain_function = 'chain_function_indicator'
                model = train_model(distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_threshold = threshold, distribution_1 = distribution_1, distribution_2 = distribution_2, **kwargs)
                save_model(model)
    
    if include_mixture_linear:
        for distr in forecast_distribution:
            distribution_1 = 'distr_trunc_normal'
            distribution_2 = distr
            distribution = 'distr_mixture_linear'
            for mean_std in mean_and_stds:
                chain_function = 'chain_function_normal_cdf'
                chain_function_mean = mean_std[0]
                chain_function_std = mean_std[1]
                model = train_model(distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_mean = chain_function_mean, chain_function_std = chain_function_std, distribution_1 = distribution_1, distribution_2 = distribution_2, **kwargs)
                save_model(model)
            
            for threshold in thresholds:
                chain_function = 'chain_function_indicator'
                model = train_model(distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, chain_function = chain_function, chain_function_threshold = threshold, distribution_1 = distribution_1, distribution_2 = distribution_2, **kwargs)
                save_model(model)



def load_model(path):
    with open(path, 'rb') as f:
        model_dict = pkl.load(f)
    
    model = EMOS(model_dict)
    return model

    

    

    