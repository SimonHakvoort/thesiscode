import string
from typing import Tuple
import numpy as np
import optuna
import tensorflow as tf

from src.models.emos import EMOS
from src.neural_networks.get_data import load_cv_data

class Objective:
    def __init__(self, feature_names_dict, objectives, twCRPS = False, train_amount = 1):
        self.twCRPS = twCRPS
        self.feature_names_dict = feature_names_dict
        self.feature_names_list = list(feature_names_dict.keys())
        print(self.feature_names_list)
        self.objectives = objectives
        self.train_amount = train_amount

        # check if objectives is correct
        for objective in objectives:
            if objective != 'CRPS' and objective[:6] != 'twCRPS':
                raise ValueError('The objective is not valid. Please use either CRPS or twCRPS')

    def get_data_i(self, i: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Preprocess the training and test data for a specific fold.

        Arguments:
            i (int): fold number

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset], with the training data as first element and the test data as the second.
        """
        train_data, test_data, data_info = load_cv_data(i, self.feature_names_dict)

        def addweight(X, y):
            return X, y, tf.constant(1, dtype=tf.float32)

        train_data = train_data.map(addweight)

        train_data = train_data.shuffle(len(train_data))

        # We remove the other features (those are only used in CNNs)
        def mapping(X, y, w):
            X_emos = {'features_emos': X['features_emos']}
            return X_emos, y, w
        
        train_data = train_data.map(mapping)

        test_data = test_data.batch(len(test_data))

        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data, test_data
    
    def compute_objective(self, emos: EMOS, objective: string, test_data: tf.data.Dataset) -> float:
        if objective == 'CRPS':
            return emos.CRPS(test_data, 90000)
        
        # check if the first 6 characters are 'twCRPS'
        elif objective[:6] == 'twCRPS':
            # get the numbers after 'twCRPS'
            twCRPS_num = objective[6:]
            return emos.twCRPS(test_data, [int(twCRPS_num)], 90000)[0]
        
    def train_emos_i(self, setup, fold, epochs, perform_batching, batch_size = None):
        train_data, test_data = self.get_data_i(fold)

        if perform_batching:
            train_data = train_data.batch(batch_size)
        else:
            train_data = train_data.batch(len(train_data))

        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        if setup['forecast_distribution'] == 'distr_mixture' or setup['forecast_distribution'] == 'distr_mixture_linear':
            distribution = setup['forecast_distribution']
            setup['forecast_distribution'] = setup['distribution_1']
            emos1 = EMOS(setup)
            emos1.fit(train_data, epochs=50, printing=False)

            setup['forecast_distribution'] = setup['distribution_2']
            emos2 = EMOS(setup)
            emos2.fit(train_data, epochs=50, printing=False)

            setup['parameters'] = {**emos1.get_parameters(), **emos2.get_parameters()}
            setup['forecast_distribution'] = distribution

        emos = EMOS(setup)

        emos.fit(train_data, epochs=epochs, printing=False)

        if setup['forecast_distribution'] == 'distr_mixture' or setup['forecast_distribution'] == 'distr_mixture_linear':
            setup.pop('parameters')

        objective_values = np.zeros(len(self.objectives))
        for i, objective in enumerate(self.objectives):
            objective_values[i] = self.compute_objective(emos, objective, test_data)

        print('Objective values for fold', fold, ':', objective_values)

        additional_metrics = {}
        if 'CRPS' not in self.objectives:
            crps = self.compute_objective(emos, 'CRPS', test_data).numpy()
            additional_metrics['CRPS'] = crps

        if 'twCRPS12' not in self.objectives:
            twcrps12 = self.compute_objective(emos, 'twCRPS12', test_data).numpy()
            additional_metrics['twCRPS12'] = twcrps12

        return objective_values, additional_metrics

    def __call__(self, trial):
        chain_function = "chain_function_normal_cdf_plus_constant"
        chain_function_mean = None
        chain_function_std = None
        chain_function_constant = None

        if self.twCRPS:
            loss = "loss_twCRPS_sample"
            chain_function_mean = trial.suggest_float('chain_function_mean', -5, 15)
            chain_function_std = trial.suggest_float('chain_function_std', 0.0001, 10, log=True)
            chain_function_constant = trial.suggest_float('chain_function_constant', 0.000001, 1, log=False)
        else:
            loss = "loss_CRPS_sample"

        samples = 250
        # optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam'])
        optimizer = 'Adam'
        # learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        learning_rate = 0.01

        forecast_distribution = trial.suggest_categorical('forecast_distribution', ['distr_trunc_normal', 
                                                                                    'distr_log_normal', 
                                                                                    'distr_gev',
                                                                                    'distr_mixture', 
                                                                                    'distr_mixture_linear'])

        if forecast_distribution == 'distr_mixture' or forecast_distribution == 'distr_mixture_linear':
            distribution_1 = 'distr_trunc_normal'
            distribution_2 = trial.suggest_categorical('distribution_2', ['distr_log_normal', 'distr_gev'])
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

        perform_batching = True# trial.suggest_categorical('perform_batching', [True, False])

        # if perform_batching:
        #     batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024])
        # else:
        #     batch_size = None
        batch_size = 256

        # epochs = trial.suggest_int('epochs', 100, 500)
        epochs = 450

        folds = [1,2,3]
        objective_values = np.zeros(len(self.objectives))
        metrics = {'CRPS': 0, 'twCRPS12': 0}
        for fold in folds:
            losses = np.zeros(len(self.objectives))
            
            for x in range(self.train_amount):
                loss, additional_metrics =  self.train_emos_i(setup, fold, epochs, perform_batching, batch_size)
                losses += loss
                trial.set_user_attr('run_' + str(x) + 'loss_fold_' + str(fold) + '_', losses.tolist())
                for key, value in additional_metrics.items():
                    metrics[key] += value / self.train_amount



            objective_values += losses / self.train_amount

        objective_values /= 3

        for key, value in metrics.items():
            trial.set_user_attr(key, value)

        for i in range(len(objective_values)):
            if objective_values[i] < 0:
                objective_values[i] = 10
            if objective_values[i] > 20:
                objective_values[i] = 20

        return objective_values.tolist()

        




