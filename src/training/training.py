import pickle as pkl

from src.models.emos import EMOS
from src.models.forecast_distributions import Mixture, TruncatedGEV, TruncatedNormal, LogNormal, GEV, Frechet, DistributionMixture, MixtureLinear, distribution_name
from src.models.get_data import get_normalized_tensor


def train_model(forecast_distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, **kwargs):
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

    data = get_normalized_tensor(neighbourhood_size, parameter_names, folds, ignore)
    X = data['X']
    y = data['y']
    variances = data['variances']
    setup['feature_mean'] = data['mean']
    setup['feature_std'] = data['std']

    setup['features'] = parameter_names
    setup["neighbourhood_size"] = neighbourhood_size

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
    

    model = EMOS(setup)

    if pretrained:
        setup['forecast_distribution'] == setup['distribution_1']
        model_1 = EMOS(setup)

        setup['forecast_distribution'] == setup['distribution_2']
        model_2 = EMOS(setup)

        model_1.fit(X, y, variances, 50, False)
        model_2.fit(X, y, variances, 50, False)
        model.set_parameters(model_1.get_parameters())
        model.set_parameters(model_2.get_parameters())
    
    if 'parameters' in kwargs:
        model.set_parameters(kwargs['parameters'])
    
    printing = False
    if 'printing' in kwargs:
        printing = kwargs['printing']



    model.fit(X, y, variances, epochs, printing)

    return model



    
def save_model(model, path = '/net/pc200239/nobackup/users/hakvoort/models/emos/'):
    if type(model.forecast_distribution) == TruncatedNormal:
        folder = 'trunc_normal/'
        distr_name = 'tn'

    if type(model.forecast_distribution) == LogNormal:
        folder = 'log_normal/'
        distr_name = 'ln'
    
    if type(model.forecast_distribution) == GEV:
        folder = 'gev/'
        distr_name = 'gev'
    
    if type(model.forecast_distribution) == Frechet:
        folder = 'frechet/'
        distr_name = 'frechet'

    if type(model.forecast_distribution) == Mixture:
        folder = 'mixture/'
        distr_name = 'mixture'

    if type(model.forecast_distribution) == MixtureLinear:
        folder = 'mixture_linear/'
        distr_name = 'mixturelinear'

    if type(model.forecast_distribution) == TruncatedGEV:
        folder = 'trunc_gev/'
        distr_name = 'trunc_gev'

    extra_name = False

    if type(model.forecast_distribution) == Mixture or type(model.forecast_distribution) == MixtureLinear:
        extra_name = True
        if type(model.forecast_distribution.distribution_1) == TruncatedNormal:
            distr1_name = 'tn'
        if type(model.forecast_distribution.distribution_1) == LogNormal:
            distr1_name = 'ln'
        if type(model.forecast_distribution.distribution_1) == GEV:
            distr1_name = 'gev'
        if type(model.forecast_distribution.distribution_1) == Frechet:
            distr1_name = 'frechet'
        if type(model.forecast_distribution.distribution_1) == TruncatedGEV:
            distr1_name = 'trunc_gev'

        if type(model.forecast_distribution.distribution_2) == TruncatedNormal:
            distr2_name = 'tn'
        if type(model.forecast_distribution.distribution_2) == LogNormal:
            distr2_name = 'ln'
        if type(model.forecast_distribution.distribution_2) == GEV:
            distr2_name = 'gev'
        if type(model.forecast_distribution.distribution_2) == Frechet:
            distr2_name = 'frechet'
        if type(model.forecast_distribution.distribution_2) == TruncatedGEV:
            distr2_name = 'trunc_gev'            

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
            chain_specs = f'mean{mean}_std{std}_constant{constant}'

    if model.loss.__name__ == 'loss_log_likelihood':
        loss = 'loglik'



    if extra_name:
        distr_name += f'_{distr1_name}_{distr2_name}'



    epochs = model.steps_made
    if epochs != 400:
        name = f'{distr_name}_{loss}_{chain_specs}_epochs{epochs}'
    else:
        name = f'{distr_name}_{loss}_{chain_specs}'

    model_dict = model.to_dict()
    with open(f'{path}{folder}{name}.pkl', 'wb') as f:
        pkl.dump(model_dict, f)

    print(f'Model saved as {path}{folder}{name}.pkl')

    
def train_and_save(forecast_distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, **kwargs):
    model = train_model(forecast_distribution, loss, optimizer, learning_rate, folds, parameter_names, neighbourhood_size, ignore, epochs, **kwargs)
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

    

    

    