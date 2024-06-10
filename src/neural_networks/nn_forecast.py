import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import register_keras_serializable

from src.neural_networks.nn_distributions import distribution_name
from src.neural_networks.nn_model import NNConvModel, NNModel, NNModel

import pdb
import pickle
import os

class NNForecast:
    def __init__(self, **kwargs):
        if not kwargs:
            return

        self.sample_size = kwargs['sample_size']
        self.features_names = kwargs['features_names']

        self.features_1d_mean = kwargs['features_1d_mean']
        self.features_1d_std = kwargs['features_1d_std']


        self._init_loss_function(**kwargs['setup_loss'])

        self.add_wind_conv = kwargs['add_wind_conv']

        if 'setup_nn_architecture' not in kwargs:
            return
        
        if self.add_wind_conv:
            self.model = NNConvModel(distribution_name(kwargs['setup_distribution']['forecast_distribution'], **kwargs['setup_distribution']), **kwargs['setup_nn_architecture'])
        else:
            self.model = NNModel(distribution_name(kwargs['setup_distribution']['forecast_distribution'], **kwargs['setup_distribution']), **kwargs['setup_nn_architecture'])


        self._init_optimizer(**kwargs['setup_optimizer'])

        if 'compile_model' in kwargs:
            if not kwargs['compile_model']:
                return

        self.metrics = []
        if 'metrics' in kwargs and kwargs['metrics'] is not None:
            if 'CRPS' in kwargs['metrics']:
                self.metrics.append(self._loss_CRPS_sample)
            if 'twCRPS_10' in kwargs['metrics']:
                self.metrics.append(self._twCRPS_10)
            if 'twCRPS_12' in kwargs['metrics']:
                self.metrics.append(self._twCRPS_12)
            if 'twCRPS_15' in kwargs['metrics']:
                self.metrics.append(self._twCRPS_15)


        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)#, run_eagerly=True)

    def _init_loss_function(self, **kwargs):
        """
        Initializes the loss function based on the provided arguments.

        Parameters:
        - kwargs (dict): Additional keyword arguments.

        Raises:
        - ValueError: If 'loss_function' is not provided or if an invalid loss function is specified.

        Returns:
        - None
        """
        if 'loss_function' not in kwargs:
            raise ValueError("loss_function must be provided")
        else:
            if kwargs['loss_function'] == 'loss_CRPS_sample':
                self.loss_function = self._loss_CRPS_sample
            elif kwargs['loss_function'] == 'loss_twCRPS_sample':
                self._init_chain_function(**kwargs)
                self.loss_function = self._loss_twCRPS_sample
            else:
                raise ValueError("Invalid loss function")
            
    def _init_optimizer(self, **kwargs):
        """
        Initializes the optimizer for the neural network.

        Args:
            optimizer (str): The optimizer to be used. Must be one of 'adam', 'sgd', or 'rmsprop'.
            learning_rate (float): The learning rate for the optimizer.

        Raises:
            ValueError: If the optimizer is not one of 'adam', 'sgd', or 'rmsprop'.

        Returns:
            None
        """
        if 'optimizer' not in kwargs:
            raise ValueError("optimizer must be provided")
        else:
            self.optimizer = kwargs['optimizer']

        if 'learning_rate' not in kwargs:
            raise ValueError("learning_rate must be provided")
        else:
            self.learning_rate = kwargs['learning_rate']
        
        if self.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")


    def _init_chain_function(self, **kwargs):
        """
        Initializes the chain function for the neural network forecast.

        Parameters:
            kwargs (dict): Keyword arguments containing the chain function and its parameters.

        Raises:
            ValueError: If the chain_function argument is not provided.
            ValueError: If the chain_function_threshold argument is not provided when using the 'chain_function_indicator' option.
            ValueError: If the chain_function_constant argument is not provided when using the 'chain_function_normal_cdf_plus_constant' option.
            ValueError: If the chain_function_mean argument is not provided when using the 'chain_function_normal_cdf' or 'chain_function_normal_cdf_plus_constant' options.
            ValueError: If the chain_function_std argument is not provided when using the 'chain_function_normal_cdf' or 'chain_function_normal_cdf_plus_constant' options.

        Returns:
            None
        """
        # Implementation code...
    def _init_chain_function(self, **kwargs):

        if 'chain_function' not in kwargs:
            raise ValueError("chain_function must be provided")
        else:
            if kwargs['chain_function'] == 'chain_function_indicator':
                self.chain_function = self._chain_function_indicator_for_twCRPS
                if 'chain_function_threshold' not in kwargs:
                    raise ValueError("chain_function_threshold must be provided")
                else:
                    self.chain_function_threshold = kwargs['chain_function_threshold']
            elif kwargs['chain_function'] in ['chain_function_normal_cdf', 'chain_function_normal_cdf_plus_constant']:
                if kwargs['chain_function'] == 'chain_function_normal_cdf':
                    self.chain_function = self._chain_function_normal_cdf_for_twCRPS
                else:
                    self.chain_function = self._chain_function_normal_cdf_plus_constant_for_twCRPS
                    if 'chain_function_constant' not in kwargs:
                        raise ValueError("chain_function_constant must be provided")
                    else:
                        self.chain_function_constant = kwargs['chain_function_constant']
                if 'chain_function_mean' not in kwargs:
                    raise ValueError("chain_function_mean must be provided")
                if 'chain_function_std' not in kwargs:
                    raise ValueError("chain_function_std must be provided")
                
                self.chain_function_normal_distribution = tfp.distributions.Normal(loc=kwargs['chain_function_mean'], scale=kwargs['chain_function_std'])

    def get_distribution(self, y_pred):
        return self.model._forecast_distribution.get_distribution(y_pred)

    def _compute_CRPS(self, y_true, y_pred, sample_size):
        distribution = self.get_distribution(y_pred)

        sample_shape = tf.concat([[sample_size], tf.ones_like(distribution.batch_shape_tensor(), dtype=tf.int32)], axis=0)

        X_1 = distribution.sample(sample_shape)
        X_2 = distribution.sample(sample_shape)

        E_1 = tf.reduce_mean(tf.abs(X_1 - tf.squeeze(y_true)), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
        
    def _loss_CRPS_sample(self, y_true, y_pred):
        return self._compute_CRPS(y_true, y_pred, self.sample_size)

    
    # def CRPS(self, X, y, sample_size):
    #     y_pred = self.predict(X)
    #     return self._compute_CRPS(y, y_pred, sample_size)
    
    def CRPS(self, dataset: tf.data.Dataset, sample_size: int) -> float:
        """
        Calculates the Continuous Ranked Probability Score (CRPS) for a given dataset.

        Args:
            dataset (tf.data.Dataset): The dataset containing input features and true labels.
            sample_size (int): The number of samples to use for prediction.

        Returns:
            float: The CRPS value.
        """
        y_true = []
        y_pred = []

        # for X, y in dataset:
        #     y_true.append(y)
        #     y_pred.append(self.predict(X))
        X, y = next(iter(dataset))
        y_pred.append(self.predict(X))
        y_true.append(y)
            
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)
        return self._compute_CRPS(y_true, y_pred, sample_size)
    
    def _chain_function_indicator(self, x, threshold):
        return tf.maximum(x, threshold)

    
    def _chain_function_indicator_for_twCRPS(self, x):
        return self.chain_function_indicator(x, self.chain_function_threshold)
    
        
    def _chain_function_normal_cdf(self, x, normal_distribution):
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part 
    
    def _chain_function_normal_cdf_for_twCRPS(self, x):
        return self._chain_function_normal_cdf(x, self.chain_function_normal_distribution)

    def _chain_function_normal_cdf_plus_constant(self, x, normal_distribution, constant):
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part + x * constant
    
    def _chain_function_normal_cdf_plus_constant_for_twCRPS(self, x):
        return self._chain_function_normal_cdf_plus_constant(x, self.chain_function_normal_distribution, self.chain_function_constant)
    
    def _twCRPS_10(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 10))
    
    def _twCRPS_12(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 12))
    
    def _twCRPS_15(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 15))

    
    def _compute_twCRPS(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_size: int, chain_function: callable) -> tf.Tensor:
        """
        Internal method that is used to compute the twCRPS for given sample size and a chaining function

        Parameters:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_size (int): The number of samples to use for estimating the twCRPS.
            chain_function (function): The chaining function to use.

        Returns:
            tf.Tensor: The twCRPS score.
        """
        # distribution is a tfp distribution object
        distribution = self.get_distribution(y_pred)

        X_1 = distribution.sample(sample_size)
        X_2 = distribution.sample(sample_size)

        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)

        E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(tf.squeeze(y_true))), axis=0)
        E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def _loss_twCRPS_sample(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Internal method that can be used as a loss function for training the model.
        It calculates the threshold-weighted Continuous Ranked Probability Score (twCRPS) for a given sample size and chain function.

        Parameters:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.

        Returns:
            tf.Tensor: The twCRPS score.
        """
        return self._compute_twCRPS(y_true, y_pred, self.sample_size, self.chain_function)
    

    def twCRPS(self, dataset: tf.data.Dataset, thresholds: list[float], sample_size: int) -> list[float]:
        """
        Calculates the threshold-weighted Continuous Ranked Probability Score (twCRPS) for a given dataset.

        Parameters:
            dataset (tf.data.Dataset): The dataset containing input features and true labels.
            thresholds (list[float]): A list of threshold values for computing twCRPS.
            sample_size (int): The number of samples to use for estimating the twCRPS.

        Returns:
            list[float]: A list of twCRPS scores corresponding to each threshold value.

        """
        y_true = []
        y_pred = []

        X, y = next(iter(dataset))
        y_pred.append(self.predict(X))
        y_true.append(y)

        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)

        scores = []
        for threshold in thresholds:
            scores.append(self._compute_twCRPS(y_true, y_pred, sample_size, lambda x: self._chain_function_indicator(x, threshold)))
        return scores
    


    def Brier_Score(self, dataset: tf.data.Dataset, thresholds: np.ndarray) -> np.ndarray:
        """
        Calculates the Brier score for a given dataset and a list of thresholds.

        Args:
            dataset (tf.data.Dataset): The dataset containing input features and true labels.
            thresholds (np.ndarray): A list of thresholds to calculate the Brier score.

        Returns:
            np.ndarray: A list of Brier scores corresponding to each threshold.
        """
        y_true = []
        y_pred = []

        X, y = next(iter(dataset))
        # for X, y in dataset:
        #     y_true.append(y)
        #     y_pred.append(self.predict(X))
        y_pred.append(self.predict(X))
        y_true.append(y)
            
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)
        distributions = self.get_distribution(y_pred)
        
        scores = np.zeros(len(thresholds))
        for i, threshold in enumerate(thresholds):
            cdf_values = distributions.cdf(threshold)
            scores[i] = tf.reduce_mean(tf.square(self.indicator_function(y_true, threshold) - cdf_values))
        return scores

    
    def indicator_function(self, y: tf.Tensor, threshold: float) -> tf.Tensor:
        """
        Applies an indicator function to the input values.

        Parameters:
            y (tf.Tensor): The input tensor.
            threshold (float): The threshold value.

        Returns:
            tf.Tensor: A tensor of the same shape as `y`, where each element is 1 if the corresponding element in `y` is less than or equal to `threshold`, and 0 otherwise.
        """
        return tf.cast(y <= threshold, tf.float32)


    @classmethod
    def my_load(cls, filepath: str, data: tf.data.Dataset) -> 'NNForecast':
        """
        Load a neural network forecast model from a file. Data is needed to compile the parameters of the model.

        Parameters:
        - filepath (str): The path to the directory containing the model files.
        - data (numpy.ndarray): The input data for prediction.

        Returns:
        - nnforecast (NNForecast): The loaded neural network forecast model.
        """
        with open(filepath + '/attributes', 'rb') as f:
            attributes = pickle.load(f)

        attributes['compile_model'] = False

        nnforecast = cls(**attributes)

        nnforecast.model.compile(optimizer=nnforecast.optimizer, loss=nnforecast.loss_function)

        nnforecast.predict(data.take(1))

        nnforecast.model.load_weights(filepath + '/model.weights.h5')

        return nnforecast

    @staticmethod
    def load_history(filepath: str) -> tf.keras.callbacks.History:
        """
        Load the training history of a neural network forecast model from a file.

        Parameters:
        - filepath (str): The path to the directory containing the history file.

        Returns:
        - history (tf.keras.callbacks.History): The training history of the model.
        """
        with open(filepath + '/history.pickle', 'rb') as f:
            history = pickle.load(f)

        return history

    
    def fit(self, dataset: tf.data.Dataset, epochs: int = 10, validation_data: tf.data.Dataset = None, early_stopping = None) -> tf.keras.callbacks.History:
        """
        Fits the neural network model to the given dataset.

        Parameters:
            dataset (tf.data.Dataset): The dataset to train the model on.
            epochs (int): The number of epochs to train the model (default: 100).
            validation_data (tf.data.Dataset): The validation dataset to evaluate the model on.

        Returns:
            history (tf.keras.callbacks.History): The history of the training process.
        """
        history = self.model.fit(dataset, epochs=epochs, validation_data=validation_data, callbacks=[early_stopping])

        return history
    

    def predict(self, X):
        return self.model.predict(X)
    
    def get_prob_distribution(self, data):
        y_pred = []
        y_true = []
        # for X, y in data:
        #     y_pred.append(self.predict(X))
        #     y_true.append(y)
        X, y = next(iter(data))
        y_pred.append(self.predict(X))
        y_true.append(y)

        y_pred = tf.concat(y_pred, axis=0)
        y_true = tf.concat(y_true, axis=0)
        return self.model._forecast_distribution.get_distribution(y_pred), y_true
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath + '/model.weights.h5')


