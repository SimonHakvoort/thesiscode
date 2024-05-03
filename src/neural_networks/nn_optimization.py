from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from bayes_opt import BayesianOptimization

from src.neural_networks.nn_forecast import NNForecast


def create_model(forecast_distribution, 
                 chain_function_mean,
                 chain_function_std,
                 chain_function_constant,
                 learning_rate,
                 conv_7x7_units,
                 conv_5x5_units,
                 conv_3x3_units,
                 number_of_hidden_layers,
                 number_of_units_in_hidden_layers,
                 dense_l2_regularization,
                 add_nwp_forecast):
    
    setup_distribution = {
        'forecast_distribution': forecast_distribution,
        'distribution_1': 'distr_trunc_normal',
        'distribution_2': 'distr_log_normal',
    }

    setup_nn_architecture = {
        'hidden_units_list': [number_of_units_in_hidden_layers] * number_of_hidden_layers,
        'dense_l1_regularization': 0.000,
        'dense_l2_regularization': dense_l2_regularization,
        'add_nwp_forecast': add_nwp_forecast,

        'conv_7x7_units': conv_7x7_units,
        'conv_5x5_units': conv_5x5_units,
        'conv_3x3_units': conv_3x3_units,
    }

    setup_loss = {
        'loss_function': 'loss_twCRPS_sample',
        'chain_function': 'chain_function_normal_cdf_plus_constant',
        'chain_function_mean': chain_function_mean,
        'chain_function_std': chain_function_std,
        'chain_function_constant': chain_function_constant,
    }

    setup_optimizer = {
        'optimizer': 'adam',
        'learning_rate': learning_rate,
    }

    setup = {
        'setup_distribution': setup_distribution,
        'setup_nn_architecture': setup_nn_architecture,
        'setup_loss': setup_loss,
        'setup_optimizer': setup_optimizer,
        'sample_size': 100,
    }

    model = NNForecast(**setup)

    return model


param_space = {
    'forecast_distribution': Categorical(['distr_trunc_normal', 'distr_log_normal', 'distr_mixture']),
    'chain_function_mean': Real(8, 20),
    'chain_function_std': Real(0.5, 5),
    'chain_function_constant': Real(0.01, 2),
    'learning_rate': Real(0.00001, 0.001),
    'conv_7x7_units': Integer(0, 7),
    'conv_5x5_units': Integer(0, 7),
    'conv_3x3_units': Integer(0, 7),
    'number_of_hidden_layers': Integer(1, 5),
    'number_of_units_in_hidden_layers': Integer(10, 200),
    'dense_l2_regularization': Real(0.0001, 0.001),
    'add_nwp_forecast': Categorical([True, False]),
}



    


