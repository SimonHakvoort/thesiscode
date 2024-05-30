import optuna
import tensorflow as tf

from src.models.emos import EMOS
from src.neural_networks.get_data import load_cv_data

class Objective:
    def __init__(self, feature_names_dict, objectives, twCRPS = False):
        self.twCRPS = twCRPS
        self.feature_names_dict = feature_names_dict
        self.feature_names_list = list(feature_names_dict.keys())
        print(self.feature_names_list)
        self.objectives = objectives

    def get_data_i(self, i):
        train_data, test_data, data_info = load_cv_data(i, self.features_names_dict)

        train_data = train_data.shuffle(len(train_data))

        train_data = train_data.batch(len(train_data))

        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        test_data = test_data.batch(len(test_data))

        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data, test_data
    

    def __call__(self, trial):
        chain_function = "chain_function_normal_cdf_plus_constant"
        chain_function_mean = None
        chain_function_std = None
        chain_function_constant = None

        if self.twCRPS:
            loss = "loss_twCRPS_sample"
            chain_function_mean = trial.suggest_uniform('chain_function_mean', -1, 15)
            chain_function_std = trial.suggest_uniform('chain_function_std', 0.001, 5)
            chain_function_constant = trial.suggest_uniform('chain_function_constant', 0.0001, 1)
        else:
            loss = "loss_CRPS_sample"

        samples = 100
        optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
        learning_rate = trial.suggest_uniform('learning_rate', 0.0001, 0.1)

        forecast_distribution = trial.suggest_categorical('forecast_distribution', ['distr_trunc_normal', 
                                                                                    'distr_log_normal', 
                                                                                    'distr_gev',
                                                                                    'distr_mixture', 
                                                                                    'distr_mixture_linear'])

        if forecast_distribution == 'distr_mixture' or forecast_distribution == 'distr_mixture_linear':
            distribution_1 = trial.suggest_categorical('distribution_1', ['distr_trunc_normal', 'distr_log_normal', 'distr_gev'])
            distribution_2 = trial.suggest_categorical('distribution_2', ['distr_trunc_normal', 'distr_log_normal', 'distr_gev'])
        else:
            distribution_1 = None
            distribution_2 = None

        random_init = False
        printing = False
        subset_size = None

        setup = {'loss': loss,
                'samples': samples, 
                'optimizer': optimizer, 
                'learning_rate': learning_rate, 
                'forecast_distribution': forecast_distribution,
                'chain_function': chain_function,
                'chain_function_mean': chain_function_mean,
                'chain_function_std': chain_function_std,
                'chain_function_constant': chain_function_constant,
                'distribution_1': distribution_1,
                'distribution_2': distribution_2,
                'random_init': random_init,
                'subset_size': subset_size,
                'printing': printing,
                'all_features': self.feature_names_list,
                'location_features': self.feature_names_list,
                'scale_features': self.feature_names_list
                    }
        
        emos = EMOS(setup)

