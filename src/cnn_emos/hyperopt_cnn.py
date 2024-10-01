import string
from typing import Tuple
import numpy as np
import optuna
import tensorflow as tf

from src.loading_data.get_data import load_cv_data
from src.cnn_emos.nn_forecast import CNNEMOS

class ObjectiveCNN:
    """
    Class used as an objective function with Optuna for hyperparameter optimization of CNNEMOS.

    This class facilitates the preprocessing of data, training of models, and computation of objective 
    functions for evaluating model performance during hyperparameter optimization.
    It trains each model on the fold for a set number of times and then takes the average performance of all models 
    over the three folds. Early stopping is used to retrieve the best parameters.
    
    This class should be used in combination with a optuna study object as objective function.
    """
    def __init__(self, feature_names_dict: dict, objectives: list, train_amount: int = 3):
        """
        Creates an instance of the ObjectiveCNN class

        Arguments:
            feature_names_dict (dict): dictionary containing the feature names and the grid sizes.
            objective (list): a list of objectives used during the optimization process.
            train_amount (int): number of times we train a model on a specific fold.
        """
        self.feature_names_dict = feature_names_dict
        self.objectives = objectives
        self.train_amount = train_amount
        self.feature_names_list = list(feature_names_dict.keys())

        # check if objectives is correct
        for objective in objectives:
            if objective != 'CRPS' and objective[:6] != 'twCRPS':
                raise ValueError('The objective is not valid. Please use either CRPS or twCRPS')


    def get_data_i(self, i: int, batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Preprocess the data for the specific fold.

        Arguments:
            i (int): fold number
            batch_size (int): batch size for the training data.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset] containing the training data (0) and the test data (1)
        """
        train_data, test_data, data_info = load_cv_data(i, self.feature_names_dict)

        train_data = train_data.shuffle(len(train_data))

        train_data = train_data.batch(batch_size)

        test_data = test_data.batch(len(test_data))

        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

        return train_data, test_data

    def compute_objective(self, nnforecast: CNNEMOS, objective: string, test_data: tf.data.Dataset) -> float:
        """
        Computes the objective values.

        Arguments:
            nnforecast (CNNEMOS): a NNForecast object.
            objective (string): a string specifying the objective.
            test_data (tf.data.Dataset): the test data.

        Returns:
            the objective value (float).
        """
        if objective == 'CRPS':
            return nnforecast.CRPS(test_data, 30000)
        # check if the first 6 characters are 'twCRPS'
        elif objective[:6] == 'twCRPS':
            # get the numbers after 'twCRPS'
            twCRPS_num = objective[6:]
            return nnforecast.twCRPS(test_data, [int(twCRPS_num)], 30000)[0]
        else:
            raise ValueError("Incorrect objective, please use CRPS or twCRPS")
        

    def train_on_fold_i(self, setup: dict, fold: int, epochs: int, batch_size: int) -> Tuple[np.ndarray, int]:
        """
        Trains a single model with a specific setup on a specific fold.

        Arguments:
            setup (dict): the setup of the NNForecast.
            fold (int): the fold on which we model is trained.
            epochs (int): the number of epochs.
            batch_size (int): the batch size.

        Returns:
            the objective values and the best epoch.
        """
        train_data, test_data = self.get_data_i(fold, batch_size)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        attempt = 0
        max_attempts = 10
        success = False

        # This was specifically for the GEV distribution, which could randomly crash. 
        # To prevent the hyperopt algorithm from stopping we kept training models until one succeeded.
        while not success and attempt < max_attempts:
            try:
                nn_forecast = CNNEMOS(**setup)
                nn_forecast.fit(train_data, epochs, test_data, early_stopping=early_stopping, verbose=0)
                success = True
            except tf.errors.InvalidArgumentError as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                attempt += 1
                if attempt >= max_attempts:
                    raise RuntimeError("Max attempts reached. Unable to train NNForecast successfully.")

        objective_values = np.zeros(len(self.objectives))
        for i, objective in enumerate(self.objectives):
            objective_values[i] = self.compute_objective(nn_forecast, objective, test_data)

        best_epoch = early_stopping.stopped_epoch - early_stopping.patience

        # In case early stopping is not activated
        if best_epoch < 0:
            best_epoch = epochs

        print("The best epoch is number: ", best_epoch)

        return objective_values, best_epoch
    
    def __call__(self, trial: optuna.Trial) -> list:
        """
        This method chooses the hyperparameter, and returns the loss(es) in a list.

        Arguments:
            trial (optuna.Trial): an trial object which select parameters.

        Returns:
            the loss values in a list.
        """
        setup = {}

        forecast_distribution = trial.suggest_categorical('Forecast Distribution', ['distr_trunc_normal', 'distr_log_normal', 'distr_mixture'])

        distribution_1 = 'distr_trunc_normal'
        distribution_2 = 'distr_log_normal'

        loss_function = 'loss_twCRPS_sample'

        chain_function = 'chain_function_normal_cdf_plus_constant'

        # Selection of the parameters from the weight function in the twCRPS.
        # For the standard deviation we use logarithmic scale.
        chain_function_mean = trial.suggest_float('cf mean', -5, 15)
        chain_function_std = trial.suggest_float('cf std', 0.0001, 10, log=True)
        chain_function_constant = trial.suggest_float('cf constant', 0.000001, 1)

        optimizer = trial.suggest_categorical('Optimizer', ['adam', 'sgd'])
        learning_rate = trial.suggest_float('Learning Rate', 0.0001, 0.03)

        dense_l2_regularization = trial.suggest_float('L2 Regularization', 0.00005, 0.1, log=True)

        number_of_layers = trial.suggest_int('Number of Layers', 1, 5)
        number_of_units = trial.suggest_int('Number of Units per Layer', 30, 200, step=10)

        hidden_units_list = [number_of_units for _ in range(number_of_layers)]

        batch_size = trial.suggest_categorical('Batch Size', [16, 32, 64, 128, 256, 512, 1024])

        conv_7x7_units = 4 # trial.suggest_int('Conv 7x7 units', 1, 5)
        conv_5x5_units = 4 # trial.suggest_int('Conv 5x5 units', 1, 5)
        conv_3x3_units = 4 # trial.suggest_int('Conv 3x3 units', 1, 5)

        sample_size = 1000

        # The maximum number of epochs will be 100.
        # The number of epochs that are used will likely be less because of early stopping.
        epochs = 100

        setup_distribution = {
            'forecast_distribution': forecast_distribution,
            'distribution_1': distribution_1,
            'distribution_2': distribution_2,
        }

        setup_nn_architecture = {
            'hidden_units_list': hidden_units_list,
            'dense_l2_regularization': dense_l2_regularization,

            'conv_7x7_units': conv_7x7_units,
            'conv_5x5_units': conv_5x5_units,
            'conv_3x3_units': conv_3x3_units,
        }

        setup_loss = {
            'loss_function': loss_function,
            'chain_function': chain_function,
            'chain_function_mean': chain_function_mean,
            'chain_function_std': chain_function_std,
            'chain_function_constant': chain_function_constant,
        }

        setup_optimizer = {
            'optimizer': optimizer,
            'learning_rate': learning_rate,
        }

        setup = {
            'setup_distribution': setup_distribution,
            'features_names': self.feature_names_list,
            'setup_loss': setup_loss,
            'setup_optimizer': setup_optimizer,
            'sample_size': sample_size,
            'setup_nn_architecture': setup_nn_architecture,
        }

        folds = [1,2,3]
        objective_values = np.zeros(len(self.objectives))

        num_epochs = []

        for fold in folds:
            losses = np.zeros(len(self.objectives))

            for iteration in range(self.train_amount):
                losses, best_epoch = self.train_on_fold_i(setup, fold, epochs, batch_size)

                print("Model ", iteration + 1, " on fold ", fold, " has loss ", losses)

                objective_values += losses

                num_epochs.append(best_epoch)

        # We want the average loss of a single model on a single fold.
        objective_values /= (3 * self.train_amount)
        
        # We also save the average number of epochs needed to train the models.
        avg_epochs = np.mean(num_epochs)

        trial.set_user_attr('Average Epochs', avg_epochs)

        return objective_values.tolist()




