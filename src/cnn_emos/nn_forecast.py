import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pickle


from src.linreg_emos.emos import BaseForecastModel
from src.cnn_emos.nn_distributions import distribution_name
from src.cnn_emos.nn_model import NNConvModel
from typing import Tuple


class CNNEMOS(BaseForecastModel):
    """
    CNNEMOS is a class designed to model the relationship between input features and distribution parameters using a Convolutional Neural Network (CNN) within the framework of an Ensemble Model Output Statistics (EMOS). 

    This class includes methods to:
    - Initialize the model with specified parameters
    - Compile and train the model
    - Predict distribution parameters
    - Calculate various performance metrics such as CRPS and Brier Score
    - Save and load model weights and training history

    Attributes:
        sample_size (int): The number of samples used for estimation.
        features_names (list): List of feature names.
        features_1d_mean (np.ndarray): Mean of the 1D features.
        features_1d_std (np.ndarray): Standard deviation of the 1D features.
        model (NNConvModel): The CNN model instance.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer for the model.
        loss_function (callable): The loss function used for training.
        metrics (list): List of metrics to monitor during training.

    Methods:
        _init_loss_function(**kwargs): Initializes the loss function based on provided arguments.
        _init_optimizer(**kwargs): Initializes the optimizer for the neural network.
        _init_chain_function(**kwargs): Initializes the chaining function for twCRPS.
        get_distribution(y_pred: tf.Tensor) -> tfp.distributions.Distribution: Returns the corresponding distribution for a prediction.
        _compute_CRPS(y_true: tf.Tensor, y_pred: tf.Tensor, sample_size: int) -> tf.Tensor: Computes the CRPS.
        _loss_CRPS_sample(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: Computes CRPS as a loss function.
        CRPS(dataset: tf.data.Dataset, sample_size: int) -> float: Calculates CRPS for a given dataset.
        _chain_function_indicator(x: tf.Tensor, threshold: tf.Tensor) -> tf.Tensor: Chaining function for an indicator.
        _chain_function_indicator_for_twCRPS(x: tf.Tensor) -> tf.Tensor: Chaining function for an indicator with a specific threshold.
        _chain_function_normal_cdf(x: tf.Tensor, normal_distribution: tfp.distributions.Normal) -> tf.Tensor: Chaining function using normal cdf.
        _chain_function_normal_cdf_for_twCRPS(x: tf.Tensor) -> tf.Tensor: Chaining function using normal cdf for twCRPS.
        _chain_function_normal_cdf_plus_constant(x: tf.Tensor, normal_distribution: tfp.distributions.Normal, constant: float) -> tf.Tensor: Chaining function using normal cdf plus constant.
        _chain_function_normal_cdf_plus_constant_for_twCRPS(x: tf.Tensor) -> tf.Tensor: Chaining function using normal cdf plus constant for twCRPS.
        _twCRPS_10(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: Estimates twCRPS at threshold 10.
        _twCRPS_12(y_true: tf.Tensor, y_pred: tf.Tensor): Estimates twCRPS at threshold 12.
        _twCRPS_15(y_true: tf.Tensor, y_pred: tf.Tensor): Estimates twCRPS at threshold 15.
        get_gev_shape(X: tf.Tensor): Returns GEV shape if applicable.
        _compute_twCRPS(y_true: tf.Tensor, y_pred: tf.Tensor, sample_size: int, chain_function: callable) -> tf.Tensor: Computes twCRPS.
        _loss_twCRPS_sample(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: Computes twCRPS as a loss function.
        twCRPS(data: tf.data.Dataset, thresholds: list[float], sample_size: int) -> list[float]: Calculates twCRPS for given thresholds.
        Brier_Score(dataset: tf.data.Dataset, probability_thresholds: np.ndarray) -> np.ndarray: Calculates Brier score for a dataset and probability_thresholds.
        indicator_function(y: tf.Tensor, threshold: float) -> tf.Tensor: Applies an indicator function to input values.
        my_load(cls, filepath: str, data: tf.data.Dataset) -> 'NNForecast': Loads a neural network forecast model from a file.
        load_history(filepath: str) -> tf.keras.callbacks.History: Loads the training history of a neural network forecast model.
        fit(dataset: tf.data.Dataset, epochs: int = 10, validation_data: tf.data.Dataset = None, early_stopping = None, steps_per_epoch = None, verbose ='auto') -> tf.keras.callbacks.History: Fits the neural network model to the given dataset.
        predict(X: dict) -> tf.Tensor: Makes a prediction and returns the distribution parameters.
        get_prob_distribution(data: tf.data.Dataset) -> Tuple[tfp.distributions.Distribution, tf.Tensor]: Returns distributions and observations for given data.
        save_weights(filepath: str) -> None: Saves the model weights to the specified filepath.
    """
    def __init__(self, **kwargs):
        if not kwargs:
            return

        self.sample_size = kwargs['sample_size']
        self.features_names = kwargs['features_names']

        if 'features_1d_mean' in kwargs:
            self.features_1d_mean = kwargs['features_1d_mean']
        else:
            self.features_1d_mean = None

        if 'features_1d_std' in kwargs:
            self.features_1d_std = kwargs['features_1d_std']
        else:
            self.features_1d_std = None


        self._init_loss_function(**kwargs['setup_loss'])

        if 'setup_nn_architecture' not in kwargs:
            return
        
        self.model = NNConvModel(distribution_name(kwargs['setup_distribution']['forecast_distribution'], **kwargs['setup_distribution']), **kwargs['setup_nn_architecture'])


        self._init_optimizer(**kwargs['setup_optimizer'])

        if 'compile_model' in kwargs:
            if not kwargs['compile_model']:
                return

        # These metrics are used to evaluate the performance on this metric during training.
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

    def _init_loss_function(self, **kwargs) -> None:
        """
        Initializes the loss function based on the provided arguments.
        In case loss_twCRPS_sample is used, we should also initialize the chaining_function.

        Parameters:
            loss_function (str): should be loss_CRPS_sample or loss_twCRPS_sample.

        Returns:
            None
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
            
    def _init_optimizer(self, **kwargs) -> None:
        """
        Initializes the optimizer for the neural network.

        Args:
            optimizer (str): The optimizer to be used. Must be one of 'adam', 'sgd', or 'rmsprop'.
            learning_rate (float): The learning rate for the optimizer.

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

    def _init_chain_function(self, **kwargs) -> None:
        """
        Initializes the chaining function.

        Arguments:
            chain_function (str): either chain_function_indicator, chain_function_normal_cdf or chain_function_normal_cdf_plus_constant.
            chain_function_threshold (float): the threshold in case chain_function_indicator is used.
            chain_function_mean (float): the mean of the Gaussian cdf in case chain_function_normal_cdf or chain_function_normal_cdf_plus_constant is used.
            chain_function_std (float): the std of the Gaussian cdf in case chain_function_normal_cdf or chain_function_normal_cdf_plus_constant is used.
            chain_function_constant (float): the constant added to the Gaussian cdf in case chain_function_normal_cdf_plus_constant is used.

        Returns:
            None.
        """
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

    def get_distribution(self, y_pred: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Returns the corresponding distribution for a prediction.

        Arguments:
            y_pred (tf.Tensor): the predicted values (which contains the mean/stds and possible weight) of the distribution.
        
        Returns:
            The probability distribution (tfp.distributions.Distribution)
        """
        return self.model._forecast_distribution.get_distribution(y_pred)

    def _compute_CRPS(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_size: int) -> tf.Tensor:
        """
        Internal method to compute the CRPS. 

        Arguments:
            y_true (tf.Tensor): the observed value.
            y_pred (tf.Tensor): the predicted values (mean, std and possible weight).
            sample_size (int): the numer of samples used to estimate expected value.

        Returns:
            an estimate of the CRPS (tf.Tensor)
        """
        # distribution is a tfp.distributions.Distribution object
        distribution = self.get_distribution(y_pred)

        sample_shape = tf.concat([[sample_size], tf.ones_like(distribution.batch_shape_tensor(), dtype=tf.int32)], axis=0)

        X_1 = distribution.sample(sample_shape)
        X_2 = distribution.sample(sample_shape)

        E_1 = tf.reduce_mean(tf.abs(X_1 - tf.squeeze(y_true)), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
        
    def _loss_CRPS_sample(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Private method that can be used as a loss function.
        It uses self.sample_size

        Arguments:
            y_true (tf.Tensor): the observed value.
            y_pred (tf.Tensor): the predicted values (mean, std and possible weight).

        Returns:
            an estimate of the CRPS (tf.Tensor)
        """
        return self._compute_CRPS(y_true, y_pred, self.sample_size)


    
    def CRPS(self, data: tf.data.Dataset, sample_size: int = 1000) -> float:
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

        X, y = next(iter(data))
        y_pred.append(self.predict(X))
        y_true.append(y)
            
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)
        return self._compute_CRPS(y_true, y_pred, sample_size).numpy()
    
    def _chain_function_indicator(self, x: tf.Tensor, threshold: tf.Tensor) -> tf.Tensor:
        """
        The chaining function of an indicator function.

        Arguments:
            x (tf.Tensor)
            threshold (tf.Tensor)

        Returns:
            the maximum of x and threshold.
        """
        return tf.maximum(x, threshold)

    
    def _chain_function_indicator_for_twCRPS(self, x: tf.Tensor) -> tf.Tensor:
        """
        The chaining function for an indicator function, where the threshold is self.chain_function_threshold.

        Arguments:
            x (tf.Tensor)

        Returns:
            the maximum of x and self.chain_function_threshold (tf.Tensor)
        """
        return self.chain_function_indicator(x, self.chain_function_threshold)
    
        
    def _chain_function_normal_cdf(self, x: tf.Tensor, normal_distribution: tfp.distributions.Normal) -> tf.Tensor:
        """
        The chaining function of a normal cdf.

        Arguments:
            x (tf.Tensor).
            normal_distribution (tfp.distribution.Normal).

        Returns:
            the chaining function with the cdf of the normal_distribution as weight function.
        """
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part 
    
    def _chain_function_normal_cdf_for_twCRPS(self, x: tf.Tensor) -> tf.Tensor:
        """
        The chaining function of a normal cdf with self.chain_function_normal_distribution.

        Arguments:
            x (tf.Tensor)

        Returns:
            The chaining function with as weight function the cdf of self.chain_function_normal_distribution.
        """
        return self._chain_function_normal_cdf(x, self.chain_function_normal_distribution)

    def _chain_function_normal_cdf_plus_constant(self, x, normal_distribution, constant):
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part + x * constant
    
    def _chain_function_normal_cdf_plus_constant_for_twCRPS(self, x):
        return self._chain_function_normal_cdf_plus_constant(x, self.chain_function_normal_distribution, self.chain_function_constant)
    
    def _twCRPS_10(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Estimates twCRPS at threshold 10, with a sample size of 1000. Can be used to monitor this quantity during training.

        Arguments:
            y_true (tf.Tensor): the observed wind speed.
            y_pred (tf.Tensor): the predicted values.

        Returns:
            An estimate of the twCRPS at threshold 10 (tf.Tensor).
        """
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 10))
    
    def _twCRPS_12(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Estimates twCRPS at threshold 12, with a sample size of 1000. Can be used to monitor this quantity during training.

        Arguments:
            y_true (tf.Tensor): the observed wind speed.
            y_pred (tf.Tensor): the predicted values.

        Returns:
            An estimate of the twCRPS at threshold 12 (tf.Tensor).
        """
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 12))
    
    def _twCRPS_15(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Estimates twCRPS at threshold 15, with a sample size of 1000. Can be used to monitor this quantity during training.

        Arguments:
            y_true (tf.Tensor): the observed wind speed.
            y_pred (tf.Tensor): the predicted values.

        Returns:
            An estimate of the twCRPS at threshold 15 (tf.Tensor).
        """
        return self._compute_twCRPS(y_true, y_pred, 1000, lambda x: self._chain_function_indicator(x, 15))
    
    def get_gev_shape(self, X: tf.Tensor):
        """
        Returns None if there is no GEV distribution in the parametric distribution. Otherwise it returns its shape.
        """
        if not self.model.has_gev():
            return None

        y_pred = self.predict(X)

        return self.model._forecast_distribution.get_gev_shape(y_pred)
        

    
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
    
    def _loss_twCRPS_sample(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
    

    def twCRPS(self, data: tf.data.Dataset, thresholds: np.ndarray, sample_size: int = 1000) -> np.ndarray:
        """
        Calculates the threshold-weighted Continuous Ranked Probability Score (twCRPS) for a given dataset.

        Parameters:
            dataset (tf.data.Dataset): The dataset containing input features and true labels.
            thresholds (np.ndarray): A list of threshold values for computing twCRPS.
            sample_size (int): The number of samples to use for estimating the twCRPS.

        Returns:
            list[float]: A list of twCRPS scores corresponding to each threshold value.
        """
        y_true = []
        y_pred = []

        X, y = next(iter(data))
        y_pred.append(self.predict(X))
        y_true.append(y)

        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)

        scores = []
        for threshold in thresholds:
            scores.append(self._compute_twCRPS(y_true, y_pred, sample_size, lambda x: self._chain_function_indicator(x, threshold)))
        return np.array(scores)
    
    
    def Brier_Score(self, data: tf.data.Dataset, probability_thresholds: np.ndarray) -> np.ndarray:
        """
        Calculates the Brier score for a given dataset and a list of thresholds.
        It computes the Brier score for a single batch.

        Args:
            data (tf.data.Dataset): The dataset containing input features and true labels.
            probability_thresholds (np.ndarray): A list of thresholds to calculate the Brier score.

        Returns:
            np.ndarray: A list of Brier scores corresponding to each threshold.
        """
        brier_scores = np.mean(self.seperate_Bier_Score(data, probability_thresholds), axis=1)

        return brier_scores
    
    def seperate_Bier_Score(self, data: tf.data.Dataset, probability_thresholds: np.ndarray) -> np.ndarray:
        """
        Similar to the Brier_Score, except that we do not take the average over the data, hence 
        the output will be a matrix.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            probability_thresholds (np.ndarray): the thresholds for the Brier score.

        Returns:
            A matrix (np.ndarray) containing the Brier score for the specified thresholds and all the stations.
        """
        X, y = next(iter(data))

        y_pred = self.predict(X)

        cdf_values = self.model._forecast_distribution.comp_cdf(y_pred, probability_thresholds)

        indicator_values = np.array([self.indicator_function(y, t) for t in probability_thresholds])

        brier_scores = (indicator_values - cdf_values) ** 2

        return brier_scores

    
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
    def my_load(cls, filepath: str, data: tf.data.Dataset) -> 'CNNEMOS':
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

    
    def fit(self, dataset: tf.data.Dataset, epochs: int = 10, validation_data: tf.data.Dataset = None, early_stopping = None, steps_per_epoch = None, verbose ='auto') -> tf.keras.callbacks.History:
        """
        Fits the neural network model to the given dataset.

        Parameters:
            dataset (tf.data.Dataset): The dataset to train the model on.
            epochs (int): The number of epochs to train the model (default: 100).
            validation_data (tf.data.Dataset): The validation dataset to evaluate the model on.
            early_stopping (tf.keras.callbacks.EarlyStopping): Early stopping callback (default: None).
            verbose ('auto', 0, 1 or 2): verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
            history (tf.keras.callbacks.History): The history of the training process.
        """
        callbacks = []
        if early_stopping is not None:
            callbacks.append(early_stopping)
            
        history = self.model.fit(dataset, epochs=epochs, validation_data=validation_data, callbacks=callbacks, steps_per_epoch=steps_per_epoch, verbose=verbose)

        return history
    

    def predict(self, X: dict) -> np.ndarray:
        """
        Makes a prediction and returns the distribution's parameters

        Arguments:
            X (dict): the dict containing the features.

        Returns:
            the predicted mean/std and possible weight of the distribution (np.ndarray).
        """
        return self.model.predict(X)
    
    def get_prob_distribution(self, data: tf.data.Dataset) -> Tuple[tfp.distributions.Distribution, tf.Tensor]:
        """
        Based on the data it returns the distributions and the observations.

        Arguments:
            data (tf.data.Dataset): data for which we compute the distribution. A single batch is taken.

        Returns:
            the distribution and the observations Tuple[tfp.distributions.Distribution, tf.Tensor].
        """
        y_pred = []
        y_true = []

        X, y = next(iter(data))
        y_pred.append(self.predict(X))
        y_true.append(y)

        y_pred = tf.concat(y_pred, axis=0)
        y_true = tf.concat(y_true, axis=0)
        return self.model._forecast_distribution.get_distribution(y_pred), y_true
    
    def save_weights(self, filepath: str) -> None:
        """
        Saves the weights of the model in filepath.
        """
        self.model.save_weights(filepath + '/model.weights.h5')


