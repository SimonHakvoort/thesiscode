import copy
import os
import pickle
from typing import Callable, List, Tuple, Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.linreg_emos.forecast_distributions import Mixture, MixtureLinear, initialize_distribution
from src.loading_data.get_data import load_cv_data
from abc import ABC, abstractmethod 
import time
tfpd = tfp.distributions

class BaseForecastModel(ABC):
    """
    Abstract base class for Climatology and the EMOS models.
    """
    @abstractmethod
    def CRPS(self, data: tf.data.Dataset, sample_size: int = 1000) -> float:
        """
        Abstract method that estimates the CRPS for a single batch in data.
        We use a sampling based approach.

        Arguments:
            data (tf.data.Dataset): data for which we estimate the CRPS.
            sample_size (int): number of samples used to estimate the CRPS per oberservation.

        Returns:
            The average CRPS over a single batch of data.
        """
        pass

    @abstractmethod
    def twCRPS(self, data: tf.data.Dataset, thresholds: np.ndarray, sample_size: int = 1000) -> np.ndarray:
        """
        Abstract method that estimates the twCRPS for a single batch in data, for all thresholds
        We use a sampling based approach. The weight function is the indicater function.

        Arguments:
            data (tf.data.Dataset): data for which we estimate the twCRPS.
            thresholds (np.ndarray): threshold at which we estimate the twCRPS
            sample_size (int): number of samples used to estimate the CRPS per oberservation.

        Returns:
            An np.ndarray with the twCRPS values.
        """
        pass

    @abstractmethod
    def Brier_Score(self, data: tf.data.Dataset, probability_thresholds: np.ndarray) -> np.ndarray:
        """
        Computes the Brier score (BS) for a single batch of data. It computes it at probability_thresholds.

        Arguments:
            data (tf.data.Dataset): data for which we compute BS.
            probability_thresholds (np.ndarray): thresholds at which we compute the BS.

        Returns:
            An np.ndarray with the BS at the thresholds.
        """
        pass

    @abstractmethod
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
        pass

class LinearEMOS(BaseForecastModel):
    """
    Class for the EMOS model with linear regression

    This class contains the EMOS model, which is used to calibrate forecasts. The model is initialized with a setup, and can be fitted to data using the fit method.
    In case we want to save the model, we can use the to_dict method to get a dictionary containing the parameters and additional settings of the model.

    To determine scores for a class that is fitted to data, we can call Brier_Score, CRPS and twCRPS.
    Most of the other methods are only used internally in the class.
    """
    def __init__(self, setup):
        """
        Initialize the EMOS model

        We initialize the model with the given setup. This is done using the getattr function.

        Arguments:
            setup: a dictionary containing the setup for the model

        The setup should contain the following keys:
            all_features: a list containing the names of all features used in the model
            loss: the loss function used to fit the model
            optimizer: the optimizer used to fit the model
            learning_rate: the learning rate of the optimizer
            forecast_distribution: the forecast distribution used in the model

        Optionally, the setup can contain the following keys:
            location_features: a list containing the names of the location features used in the model, used to determine the mean
            scale_features: a list containing the names of the scale features used in the model, used to determine the standard deviation
            neighbourhood_size: the size of the neighbourhood used in the model to determine the spatial variance
            feature_mean: the mean of the features used in the model
            feature_std: the standard deviation of the features used in the model
            chain_function: the chain function used in the model, with the corresponding as separate keys
            samples: the amount of samples used in the loss function in case we use a sample based loss function
        """
        self.all_features = copy.deepcopy(setup['all_features'])

        if 'location_features' in setup:
            self.location_features = copy.deepcopy(setup['location_features'])
        else:
            self.location_features = copy.deepcopy(setup['all_features'])

        if 'scale_features' in setup:
            self.scale_features = copy.deepcopy(setup['scale_features'])
        else:
            self.scale_features = copy.deepcopy(setup['all_features'])
        

        self.num_features = len(self.all_features)
        
        # neighbourhood_size is used in case we use spatial variance.
        if 'neighbourhood_size' not in setup:
            self.neighbourhood_size = None
        else:
            self.neighbourhood_size = setup['neighbourhood_size']

        self._init_loss(setup)
        
        if self.need_chain:
            self._init_chain_function(setup)

        self._init_optimizer(setup)
        
        self._init_forecast_distribution(setup)

        # The history of the optimization process containing the losses.
        # This will only be used when we load a saved class in.
        self.hist = []
        if 'history' in setup:
            self.hist = setup['history']


        # Optionally we can initialize the feature mean and standard deviation with the given values. Not sure whether this needs to be included
        if 'feature_mean' in setup and 'feature_std' in setup:
            if setup['feature_mean'] is not None and setup['feature_std'] is not None:
                self.feature_mean = tf.Variable(setup['feature_mean'])
                self.feature_std = tf.Variable(setup['feature_std'])
        else:
            self.feature_mean = None
            self.feature_std = None

        # Optionally we can initialize the amount of steps made with the optimizer
        if 'steps_made' in setup:
            self.steps_made = setup['steps_made']
        else:
            self.steps_made = 0

        # to save on which fold the model is trained.
        if 'folds' in setup:
            self.folds = setup['folds']

    def _init_loss(self, setup: dict) -> None:
        """
        Setup the loss function of the model.

        Arguments:
            setup (dict): a dictionary containing the setup for the model.

        The setup should contain the following keys:
            loss: the loss function used to fit the model
            samples: the amount of samples used in the loss function in case we use a sample based loss function
        """
        self.need_chain = False
        try:
            # We set the loss to either CRPS or twCRPS
            self.loss = getattr(self, setup['loss'])
            if self.loss == self.loss_CRPS_sample or self.loss == self.loss_twCRPS_sample:
                if 'samples' not in setup:
                    raise ValueError("Amount of samples not specified")
                else:
                    self.samples = setup['samples']
                if self.loss == self.loss_twCRPS_sample:
                    self.need_chain = True
        except AttributeError:
            raise ValueError("Invalid loss function: " + setup['loss'])   

    def _init_chain_function(self, setup: dict) -> None:
        """
        Setup the chaining function of the model.

        Arguments:
            setup (dict): a dictionary containing the setup for the model.

        The setup should contain the following keys:
            chain_function: the selected chaining function.
            chain_function_threshold (optional): the threshold in case of indicator weight function.
            chain_function_mean (optional): the mean of the Gaussian cdf.
            chain_function_std (optional): the std of the Gaussian cdf.
            chain_function_constant (optional): the constant added to the Gaussian cdf.
        """
        try:
            if setup['chain_function'] == 'chain_function_indicator':
                self.chain_function = self.chain_function_indicator
                if 'chain_function_threshold' not in setup:
                    raise ValueError("Threshold of the chain function not specified")
                else:
                    self.chain_function_threshold = tf.constant(setup['chain_function_threshold'], dtype=tf.float32)
            elif setup['chain_function'] == 'chain_function_normal_cdf':
                self.chain_function = self.chain_function_normal_cdf
                if 'chain_function_mean' not in setup:
                    raise ValueError("Mean  of the chain function not specified")
                else:
                    self.chain_function_mean = tf.constant(setup['chain_function_mean'], dtype=tf.float32)
                if 'chain_function_std' not in setup:
                    raise ValueError("Standard deviation of the chain function not specified")
                else:
                    self.chain_function_std = tf.constant(setup['chain_function_std'], dtype=tf.float32)
                self.chain_normal_distr = tfpd.Normal(self.chain_function_mean, self.chain_function_std)
            elif setup['chain_function'] == 'chain_function_normal_cdf_plus_constant':
                self.chain_function = self.chain_function_normal_cdf_plus_constant
                if 'chain_function_mean' not in setup:
                    raise ValueError("Mean  of the chain function not specified")
                else:
                    self.chain_function_mean = tf.constant(setup['chain_function_mean'], dtype=tf.float32)
                if 'chain_function_std' not in setup:
                    raise ValueError("Standard deviation of the chain function not specified")
                else:
                    self.chain_function_std = tf.constant(setup['chain_function_std'], dtype=tf.float32)
                self.chain_normal_distr = tfpd.Normal(self.chain_function_mean, self.chain_function_std)
                if 'chain_function_constant' not in setup:
                    raise ValueError("Constant of the chain function not specified")
                else:
                    self.chain_function_constant = tf.constant(setup['chain_function_constant'], dtype=tf.float32)
        except AttributeError:
            raise ValueError("Invalid chain function: " + setup['chain_function'])  

    def _init_optimizer(self, setup: dict) -> None:
        """
        Initialization of the optimizer. 

        Arguments:
            setup (dict): dictionary containing the setup of the model.

        The setup should contain the following keys:
            learning_rate: the learning rate of the optimizer.
            optimizer: the name of a tf.optimizer.
        """
        try:
            if 'learning_rate' not in setup:
                raise ValueError("Learning rate not specified")
            learning_rate = round(float(setup['learning_rate']), 6)
            self.optimizer = getattr(tf.optimizers, setup['optimizer'])(learning_rate=learning_rate)
        except AttributeError:
            raise ValueError("Invalid optimizer: " + setup['optimizer']) 
        
    def _init_forecast_distribution(self, setup: dict) -> None:
        """
        Initialization of the parametric distribution of the probabilistic forecast.

        Arguments:
            setup (dict): dictionary containing the setup of the model.

        The setup should contain the following keys:
            forecast_distribution (str): name of the forecast distribution of the model.
            location_features (list): a list containing the names of the features used for the location (mu).
            scale_features (list): a list containing the names of the features used for the scale (sigma).
            all_features (list): a list containing all the names of the features (should be sorted in the same way as the data).
            parameters (optional, dict): parameters in case we pre-train models.
            random_init (optional, bool): boolean stating whether random initialization should be used.
            distribution_1 (optional, str): name of the first distribution in case a mixture distribution is used.
            distribution_2 (optional, str): name of the second distribution in case a mixture distribution is used.
        """
        if "parameters" in setup:
            parameters = setup["parameters"]
        else:
            parameters = {}

        if "forecast_distribution" not in setup:
            raise ValueError("Forecast distribution not specified")

        if "location_features" not in setup:
            raise ValueError("Location features not specified")
        
        if "scale_features" not in setup:
            raise ValueError("Scale features not specified")

        if "all_features" not in setup:
            raise ValueError("All features not specified")

        random_init = False
        if "random_init" in setup:
            random_init = setup["random_init"]

        distribution_1 = None
        distribution_2 = None

        if setup['forecast_distribution'] == 'distr_mixture' or setup['forecast_distribution'] == 'distr_mixture_linear':
            if 'distribution_1' not in setup or 'distribution_2' not in setup:
                raise ValueError("Please specify the distributions for the mixture")
            else:
                distribution_1 = setup['distribution_1']
                distribution_2 = setup['distribution_2']
        
        self.forecast_distribution = initialize_distribution(
            setup['forecast_distribution'], 
            setup["all_features"], 
            setup["location_features"],
            setup["scale_features"],
            parameters, 
            random_init,
            distribution_1,
            distribution_2)        
        
    
    def __str__(self) -> str:
        """
        Returns all the information regarding the setup of the model.
        """
        # Loss function info
        loss_info = f"Loss function: {self.loss.__name__}"
        if hasattr(self, 'samples'):
            loss_info += f" (Samples: {self.samples})"

        # Optimizer info
        optimizer_info = f"Optimizer: {self.optimizer.__class__.__name__}"
        learning_rate_info = f"Learning rate: {self.optimizer.learning_rate.numpy()}"

        # Forecast distribution info
        forecast_distribution_info = f"Forecast distribution: {self.forecast_distribution.name()}"

        # Feature info
        feature_info = f"Features: {', '.join(self.all_features)}"
        location_features = f"Location features: {', '.join(self.location_features)}"
        scale_features = f"Scale features: {', '.join(self.scale_features)}"
        num_features_info = f"Number of features: {self.num_features}"
        neighbourhood_size_info = f"Neighbourhood size: {self.neighbourhood_size}"

        # Parameters info
        parameters_info = "Parameters:"
        for parameter, value in self.forecast_distribution.parameters.items():
            parameters_info += f"\n  {parameter}: {value}"

        # Chaining function info
        chaining_function_info = ""
        if hasattr(self, 'chain_function'):
            chaining_function_info = f"Chaining function: {self.chain_function.__name__}"
            if hasattr(self, 'chain_function_threshold'):
                chaining_function_info += f" (Threshold: {self.chain_function_threshold.numpy()})"
            elif hasattr(self, 'chain_function_mean') and hasattr(self, 'chain_function_std'):
                chaining_function_info += f" (Mean: {self.chain_function_mean.numpy()}, Std: {self.chain_function_std.numpy()})"
                if hasattr(self, 'chain_function_constant'):
                    chaining_function_info += f", Constant: {self.chain_function_constant.numpy()}"

        # Extra info for mixtrue distributions
        distribution_info = ""
        if type(self.forecast_distribution) == Mixture:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            distribution_info += f"Mixture weight: {self.forecast_distribution.get_weight()}\n"
        elif type(self.forecast_distribution) == MixtureLinear:
            distribution_info = f"Distribution 1: {self.forecast_distribution.distribution_1.name()}\n"
            distribution_info += f"Distribution 2: {self.forecast_distribution.distribution_2.name()}\n"
            weight_a, weight_b = self.forecast_distribution.get_weights()
            distribution_info += f"Mixture weight a: {weight_a}\n"
            distribution_info += f"Mixture weight b: {weight_b}\n"

        return (
            f"Linear EMOS Model Information:\n"
            f"{loss_info}\n"
            f"{forecast_distribution_info}\n"
            f"{distribution_info}"
            f"{parameters_info}\n"
            f"{feature_info}\n"
            f"{location_features}\n"
            f"{scale_features}\n"
            f"{num_features_info}\n"
            f"{neighbourhood_size_info}\n"
            f"{chaining_function_info}\n"
            f"{optimizer_info}\n"
            f"{learning_rate_info}\n"
        )

    
    def get_parameters(self) -> dict:
        """
        Return the parameters as np.arrays of the model in a dictionary.

        Returns:
            A dictionary containing the parameters of the model.
        """
        return self.forecast_distribution.parameters
    
    def set_parameters(self, parameters) -> None:
        """
        Set the parameters of the model to the given values.

        Arguments:
            parameters: a dictionary containing the parameters of the model.
        """
        self.forecast_distribution.parameters = parameters
    
    def to_dict(self) -> dict:
        """
        Return the model as a dictionary, including the settings and parameters.
        Can be used to save the settings and parameters of the model and load it back in.

        Returns:
            dictionary containing the setup for the class and the parameters.
        """
        model_dict = {
            'loss' : self.loss.__name__,
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.learning_rate.numpy(),
            'forecast_distribution': self.forecast_distribution.name(),
            'feature_mean': self.feature_mean.numpy() if self.feature_mean is not None else None,
            'feature_std': self.feature_std.numpy() if self.feature_std is not None else None,
            'steps_made': self.steps_made,
            'all_features': self.all_features,
            'location_features': self.location_features,
            'scale_features': self.scale_features,
            'neighbourhood_size': self.neighbourhood_size
        }
        model_dict['parameters'] = self.get_parameters()

        if hasattr(self, 'samples'):
            model_dict['samples'] = self.samples

        if self.need_chain:
            if self.chain_function == self.chain_function_indicator:
                model_dict['chain_function'] = 'chain_function_indicator'
                model_dict['chain_function_threshold'] = self.chain_function_threshold.numpy()
            elif self.chain_function == self.chain_function_normal_cdf:
                model_dict['chain_function'] = 'chain_function_normal_cdf'
                model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
                model_dict['chain_function_std'] = self.chain_function_std.numpy()
            elif self.chain_function == self.chain_function_normal_cdf_plus_constant:
                model_dict['chain_function'] = 'chain_function_normal_cdf_plus_constant'
                model_dict['chain_function_mean'] = self.chain_function_mean.numpy()
                model_dict['chain_function_std'] = self.chain_function_std.numpy()
                model_dict['chain_function_constant'] = self.chain_function_constant.numpy()

        if type(self.forecast_distribution) == Mixture or type(self.forecast_distribution) == MixtureLinear:
            model_dict['distribution_1'] = self.forecast_distribution.distribution_1.name()
            model_dict['distribution_2'] = self.forecast_distribution.distribution_2.name()

        model_dict['history'] = self.hist

        return model_dict
    
    def get_gev_shape(self, X: tf.Tensor) -> Union[np.ndarray, None]:
        """
        Returns the shape of the GEV distribution in case this is part of the parametric distribution.
        The output has the same first dimension (sample size) as X.

        Arguments:
            X (tf.Tensor): a Tensor (features_emos) for which we want to compute the GEV shape.

        Returns:
            The GEV shape or None.
        """
        gev_shape = self.forecast_distribution.get_gev_shape()

        if gev_shape is None:
            return None
        
        return np.full(X['features_emos'].shape[0], gev_shape)
    
    def normalize_features(self, X):
        """
        Normalize the features of the model.

        Arguments:
            X: the input data of shape (n, m), where n is the number of samples and m is the number of features.

        Returns:
            the normalized input data.
        """
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Feature mean and standard deviation not set")
        return (X - self.feature_mean) / self.feature_std

    
    def indicator_function(self, y, t):
        """
        The indicator function, which returns 1 if y <= t, and 0 otherwise.

        Arguments:
            y: the input value.
            t: the threshold.

        Returns:
            1 if y <= t, 0 otherwise.
        """
        return tf.cast(y <= t, tf.float32)
 
    def loss_log_likelihood(self, X, y):
        """
        The loss fuction for the log likelihood, based on the forecast distribution and observations.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            y (tf.Tensor): the observations of shape (n,).

        Returns:
        - the loss value.
        """
        forecast_distribution = self.forecast_distribution.get_distribution(X)
        return -tf.reduce_mean(forecast_distribution.log_prob(y))
    
    def get_prob_distribution(self, data: tf.data.Dataset) -> Tuple[tfp.distributions.Distribution, tf.Tensor]:
        """
        Returns the distribution and the observations a single batch taken from data.

        Arguments:
            data (tf.data.Dataset): the data for which we want to compute the distribution

        Returns:
            The distribution and the observations.
        """
        X, y = next(iter(data))
        distributions = self.forecast_distribution.get_distribution(X['features_emos'])

        return distributions, y
    
    def CRPS(self, data: tf.data.Dataset, sample_size: int = 1000) -> float:
        """
        Estimates the CRPS for the given data, using the specified sample size

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            sample_size (int): the amount of samples used to estimate the expected value of the CRPS.

        Returns:
            the CRPS value (float).
        """
        X, y = next(iter(data))
        total_crps = tf.reduce_mean(self._crps_computation(X['features_emos'], y, sample_size)).numpy()
        return total_crps

    
    def _crps_computation(self, X: tf.Tensor, y: tf.Tensor, samples: int) -> tf.Tensor:
        """
        The computation of the CRPS, based on the features and observations.
        We use a sample based approach to estimate the expected value of the CRPS.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            y (tf.Tensor): the observations of shape (n,).
            samples (int): the amount of samples used to estimate the expected value of the CRPS.

        Returns:
            a tf.Tensor of shape (n,) containing the losses for all the samples.
        """
        forecast_distribution = self.forecast_distribution.get_distribution(X)

        #X_1 has shape (samples, n), where n is the number of observations
        X_1 = forecast_distribution.sample(samples)
        X_2 = forecast_distribution.sample(samples)

        # y will be broadcasted to the shape of X_1.
        # We then take the mean over the samples, so E_1 and E_2 have shape (n,)
        E_1 = tf.reduce_mean(tf.abs(X_1 - y), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return E_1 - 0.5 * E_2
    
    def loss_cPIT(self, X: tf.Tensor, y: tf.Tensor):
        """
        Loss function that can be used where we minimize the cPIT scores for threshold 0, 5, 10 and 15.

        Arguments:
            X (tf.Tensor): the features.
            y (tf.Tensor): the observations.

        Returns:
            The average cPIT score over the thresdholds.
        """
        return tf.reduce_mean([self.calc_cPIT(X, y, t) for t in [0, 5, 10, 15]])
    
    def calc_cPIT(self, X: tf.Tensor, y: tf.Tensor, t: float) -> tf.Tensor:
        """
        Function used for computing cPIT scores.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            y (tf.Tensor): the observations of shape (n,).
            t (float): threshold for which the cPIT score should be computed.

        Returns:
            tf.Tensor containing the cPIT scores.
        """
        indices = tf.where(y > t)
        indices = tf.reshape(indices, [-1])

        y_greater = tf.gather(y, indices)
        X_greater = tf.gather(X, indices)

        cdf = self.forecast_distribution.get_distribution(X_greater).cdf

        if t == 0:
            probabilities = cdf(y_greater)
        elif t > 0:
            upper = cdf(y_greater) - cdf(t)
            lower = 1 - cdf(t)
            # remove the points where lower is 0
            mask = tf.where(lower == 0, False, True)
            upper = tf.boolean_mask(upper, mask)
            lower = tf.boolean_mask(lower, mask)
            probabilities = upper / lower

        probabilities = tf.sort(probabilities)

        return tf.reduce_mean(tf.abs(probabilities - tf.linspace(0.0, 1.0, tf.shape(probabilities)[0])))

    def loss_CRPS_sample(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Function that is used in the constructor to set as loss function, with sample size self.samples.

        Arguments:
            X (tf.Tensor): input features
            y (tf.Tensor): observations
        """
        return self._crps_computation(X, y, self.samples)
        
    
    def Brier_Score(self, data: tf.data.Dataset, probability_thresholds: np.ndarray) -> np.ndarray:
        """
        The loss function for the Brier score, based on the forecast distribution and observations, using a tf.dataset.
        We compute the Brier score for a single batch of the dataset, meaning that in case we want to compute the Brier score over the entire
        dataset, we should put it in a single batch.

        Arguments:
            data (tf.dataset): the dataset containing the input data and observations.
            probability_thresholds (np.ndarray): the threshold for the Brier score.

        Returns:
            An np.ndarray containing the Brier scores for the specified thresholds.
        """
        # Calculate the Brier scores vectorized using seperate_Brier_Score and then take the mean over all 
        # samples that have the same threshold. This will give the average Brier score for that specific threshold.
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
            A matrix (np.ndarray) containing the Brier score for the specified thresholds and all the samples
        """
        X, y = next(iter(data))

        # Predict the CDF values for all thresholds
        cdfs = self.forecast_distribution.comp_cdf(X['features_emos'], probability_thresholds)  

        # Compute the indicator values for all thresholds
        indicator_matrix = np.array([self.indicator_function(y, t) for t in probability_thresholds])

        # Calculate the Brier scores vectorized
        brier_scores = (indicator_matrix - cdfs) ** 2

        return brier_scores
    
    def twCRPS(self, data: tf.data.Dataset, thresholds: np.ndarray, sample_size: int) -> np.ndarray:
        """
        The loss function for the twCRPS, based on the forecast distribution and observations, using a tf.dataset.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            thresholds (np.ndarray): the thresholds for the twCRPS.
            sample_size (int): the amount of samples used to estimate the expected value of the twCRPS.

        Returns:
            the twCRPS at the given thresholds (np.ndarray).
        """
        X, y = next(iter(data))
        forecast_distribution = self.forecast_distribution.get_distribution(X['features_emos'])
        twcrps = np.zeros(len(thresholds))

        for i, threshold in enumerate(thresholds):
            chain_function = lambda x: self.chain_function_indicator_general(x, threshold)
            X_1 = forecast_distribution.sample(sample_size)
            X_2 = forecast_distribution.sample(sample_size)
            vX_1 = chain_function(X_1)
            vX_2 = chain_function(X_2)
            E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(y)), axis=0)
            E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)
            twcrps[i] = tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)

        return twcrps

    
    def loss_twCRPS_sample(self, X: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Internal method that can be used as loss function to train the LinearEMOS model with the twCRPS function.
        As chaining function we use self.chain_function and as sample size we use self.samples.

        Arguments:
            X (tf.Tensor): features.
            y (tf.Tensor): The observations.

        Returns:
            The twCRPS as a tf.Tensor
        """
        return self.loss_twCRPS_sample_general(X, y, self.chain_function, self.samples)
        
    def loss_twCRPS_sample_general(self, X: tf.Tensor, y: tf.Tensor, chain_function: Callable[[tf.Tensor], tf.Tensor], samples: int) -> tf.Tensor:
        """
        The loss function for the twCRPS, based on the forecast distribution and observations. 
        We use a sample based approach to estimate the expected value of the twCRPS.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features
            y (tf.Tensor): the observations of shape (n,)
            chain_function (callable): the function used to compute the chain function
            samples (int): the amount of samples used to estimate the expected value of the twCRPS

        Returns:
            a tf.Tensor of shape (n,) containing the losses for all the samples.
        """	
        forecast_distribution = self.forecast_distribution.get_distribution(X)

        #X_1 has shape (samples, n), where n is the number of observations
        X_1 = forecast_distribution.sample(samples)
        X_2 = forecast_distribution.sample(samples)



        # Transform X_2 and X_2 with the specified chaining function.
        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)

        # y will be broadcasted to the shape of X_1.
        # We then take the mean over the samples, so E_1 and E_2 have shape (n,)
        E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(y)), axis=0)
        E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

        return E_1 - 0.5 * E_2
    
        
    def chain_function_indicator(self, y):
        """
        Implements the chain function in case the weight function is the indicator function, which is used in weighted loss functions.

        Arguments:
        - y: the input value.

        Returns:
        - the maximum of y and the threshold.
        """
        return self.chain_function_indicator_general(y, self.chain_function_threshold)
    
    def chain_function_indicator_general(self, y, threshold):
        return tf.maximum(y, threshold)
    
    def chain_function_normal_cdf(self, y: tf.Tensor) -> tf.Tensor:
        """
        Implements the chain function in case the weight function is the normal cumulative distribution, with mean and standard deviation in the setup.

        Arguments:
            y: the input value.

        Returns:
            (y - mean) * cdf(y) + std^2 * pdf(y)
        """
        first_part = (y - self.chain_function_mean) * self.chain_normal_distr.cdf(y)
        second_part = self.chain_function_std ** 2 * self.chain_normal_distr.prob(y)
        return first_part + second_part  
    
    def chain_function_normal_cdf_plus_constant(self, y: tf.Tensor) -> tf.Tensor:
        """
        Implements the chain function in case the weight function is the normal cumulative distribution, with mean and standard deviation in the setup, and a constant.

        Arguments:
            y (tf.Tensor): the input value.

        Returns:
            (y - mean) * cdf(y) + std^2 * pdf(y) + constant * y
        """
        first_part = (y - self.chain_function_mean) * self.chain_normal_distr.cdf(y)
        second_part = self.chain_function_std ** 2 * self.chain_normal_distr.prob(y)
        return first_part + second_part + self.chain_function_constant * y

    def _compute_loss_and_gradient(self, X: tf.Tensor, y: tf.Tensor, w: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Compute the loss and the gradient of the loss with respect to the parameters of the model, 
        which are the parameters of the forecast distribution. 

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            y (tf.Tensor): the observations of shape (n,).
            w (tf.Tensor): the weights of the samples with shape (n,)

        Returns:
            loss_value: the weighted loss value.
            grads: the gradients of the loss with respect to the parameters of the model.
        """
        with tf.GradientTape() as tape:
            # Compute the losses for all the samples and multiply this with the corresponding weights.
            weighted_loss = self.loss(X, y) * w
            loss_value = tf.reduce_mean(weighted_loss)
        
        grads = tape.gradient(loss_value, [*self.forecast_distribution.parameter_dict.values()])
        return loss_value, grads
    
    @tf.function
    def _train_step(self, X: tf.Tensor, y: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        """
        Perform a training step with the optimizer.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.
            y (tf.Tensor): the observations of shape (n,).
            w (tf.Tensor): the weights of shape (n,)

        Returns:
            the loss value.
        """
        loss_value, grads = self._compute_loss_and_gradient(X, y, w)

        # This checks whether nan is inside the gradients
        if tf.math.reduce_any(tf.math.is_nan(grads[0])):
            return -1.0
        
        # Clip the gradients for stability
        clipped_grads = [tf.clip_by_value(grad, -1.0, 1.0) for grad in grads]

        # Apply the gradients to the parameters that are inside the forecast_distribution
        self.optimizer.apply_gradients(zip(clipped_grads, self.forecast_distribution.parameter_dict.values()))
        return loss_value
    
    def predict(self, X: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Predict the forecast distribution for the given input data.

        Arguments:
            X (tf.Tensor): the input data of shape (n, m), where n is the number of samples and m is the number of features.

        Returns:
            the forecast distribution.
        """
        return self.forecast_distribution.get_distribution(X)

    
    def fit(self, data: tf.data.Dataset, epochs: int, printing: bool = True, validation_data: tf.data.Dataset = None) -> dict:
        """
        Fit EMOS with linear regression to the given data.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data (X), the observations (y) and the corresponding weights (w).
            epochs (int): the amount of epochs to train the model.
            printing (bool): whether to print the loss value at each epoch.
            validation_data (tf.data.Dataset): dataset on which the validation is performed at the end of every epoch.

        Returns:
            a dictionary with two keys, hist which contains the history of losses per epoch, and time_hist, which contains the losses at intervals of 0.1 seconds.
        """
        validation_loss = {}

        if validation_data is not None:
            X_val, y_val = next(iter(validation_data))
            X_val = X_val['features_emos']
            val_loss = self.loss(X_val, y_val)
            validation_loss[0] = total_loss = tf.reduce_mean(val_loss).numpy()


        start = time.time()
        time_losses = 0
        interval = 0.5

        num_batches = 0

        for epoch in range(epochs):
            epoch_losses = 0.0
            batch_count = 0.0
            for X, y, w in data:
                loss_value = self._train_step(X['features_emos'], y, w)

                # Save the loss value for each epoch
                epoch_losses += loss_value
                batch_count += 1.0

                # Save the loss value for each time interval
                time_losses += loss_value
                current_time = time.time()

                if validation_data is not None:
                    # Check wheter more than 0.2 second has passed
                    if current_time - start > interval and num_batches > 1:
                        val_loss = self.loss(X_val, y_val)
                        total_loss = tf.reduce_mean(val_loss).numpy()
                        validation_loss[current_time - start] = total_loss
                        interval += 0.5
                        num_batches = 0
                    else:
                        num_batches += 1

            epoch_mean_loss = epoch_losses / batch_count
            self.hist.append(epoch_mean_loss)
            if printing and epoch % 1 == 0:
                tf.print("Epoch: ", epoch, " Loss: ", epoch_mean_loss)

        output_dict = {'hist': self.hist, 'validation_loss': validation_loss}
        return output_dict
                
            





### Can be used as example for bagging estimator for CNNs
class BootstrapEmos():

    def __init__(self, setup, filepath, epochs, batch_size, cv_number, features_names_dict):
        self.setup = setup
        self.filepath = filepath
        self.num_models = 0

        self.epochs = epochs
        self.batch_size = batch_size
        self.cv_number = cv_number
        self.features_names_dict = features_names_dict
        self.models = None

    def save_bootstrap_info(self):
        info = {'setup': self.setup,
                'num_models': self.num_models,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'cv_number': self.cv_number,
                'features_names_dict': self.features_names_dict,
                'filepath': self.filepath}
        
        with open(os.path.join(self.filepath, 'bootstrap_info.pkl'), 'wb') as f:
            pickle.dump(info, f)

    @classmethod 
    def load(cls, filepath):
        with open(os.path.join(filepath, 'bootstrap_info.pkl'), 'rb') as f:
            info = pickle.load(f)
        
        instance = cls(
            setup = info['setup'],
            filepath = info['filepath'],
            epochs = info['epochs'],
            batch_size = info['batch_size'],
            cv_number = info['cv_number'],
            features_names_dict = info['features_names_dict']
        )

        instance.num_models = info['num_models']

        return instance

 
    def train_models(self, number):
        train_data, test_data, data_info = load_cv_data(self.cv_number, self.features_names_dict)
        #train_data_list = list(train_data.as_numpy_iterator())
        train_data = train_data.batch(len(train_data))

        X, y = next(iter(train_data))

        amount_of_data = y.shape[0]

        for _ in range(number):
            model = LinearEMOS(self.setup)
            bootstrap_sample = self._make_bootstrap_sample(X, y)

            bootstrap_sample = bootstrap_sample.shuffle(amount_of_data)

            if self.batch_size is not None:
                bootstrap_sample = bootstrap_sample.batch(self.batch_size)
            else:
                bootstrap_sample = bootstrap_sample.batch(amount_of_data)

            bootstrap_sample = bootstrap_sample.prefetch(tf.data.experimental.AUTOTUNE)



            model.fit(bootstrap_sample, self.epochs, printing=True)
            
            model_dict = model.to_dict()
            name = 'model_' + str(self.num_models) + '.pkl'
            dir_path = os.path.join(self.filepath, 'models')
            os.makedirs(dir_path, exist_ok=True)
            path = os.path.join(dir_path, name)
            with open(path, 'wb') as f:
                pickle.dump(model_dict, f)

            self.num_models += 1
            print("Model ", self.num_models, " trained")

    def load_models(self):
        models = []
        for i in range(self.num_models):
            name = 'model_' + str(i) + '.pkl'
            path = os.path.join(self.filepath, 'models', name)
            with open(path, 'rb') as f:
                model_dict = pickle.load(f)
            model = LinearEMOS(model_dict)
            models.append(model)
        self.models = models
    

    def _make_bootstrap_sample(self, X, y):
        X_f = X['features_emos']

        indices = np.random.choice(X_f.shape[0], X_f.shape[0], replace=True)

        X_f = tf.gather(X_f, indices)

        y = tf.gather(y, indices)

        dataset = tf.data.Dataset.from_tensor_slices((X_f, y))

        def mapping(X, y):
            return {'features_emos': X}, y
        
        dataset = dataset.map(mapping)

        return dataset
    
    def retrain_model_i(self, i):
        train_data, test_data, data_info = load_cv_data(self.cv_number, self.features_names_dict)
        train_data = train_data.batch(len(train_data))

        X, y = next(iter(train_data))

        amount_of_data = y.shape[0]

        model = LinearEMOS(self.setup)

        bootstrap_sample = self._make_bootstrap_sample(X, y)

        bootstrap_sample = bootstrap_sample.shuffle(amount_of_data)

        if self.batch_size is not None:
            bootstrap_sample = bootstrap_sample.batch(self.batch_size)
        else:
            bootstrap_sample = bootstrap_sample.batch(amount_of_data)

        bootstrap_sample = bootstrap_sample.prefetch(tf.data.experimental.AUTOTUNE)

        model.fit(bootstrap_sample, self.epochs, printing=True)

        model_dict = model.to_dict()
        name = 'model_' + str(i) + '.pkl'
        path = os.path.join(self.filepath, 'models', name)
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
    
    def Brier_Score(self, data: tf.data.Dataset, values: np.ndarray) -> np.ndarray:
        """
        Computes the Brier score for the given data and values.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            values (np.ndarray): the values for the Brier score.

        Returns:
            the Brier score values for the given values.
        """
        if self.models is None:
            self.load_models()

        brier_scores = np.zeros(shape=(len(values), self.num_models))
        for i, model in enumerate(self.models):
            brier_scores[:, i] = model.Brier_Score(data, values)

        return brier_scores
    
    def twCRPS(self, data: tf.data.Dataset, thresholds: np.ndarray, samples: int) -> np.ndarray:
        """
        Computes the twCRPS for the given data and thresholds.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            thresholds (np.ndarray): the thresholds for the twCRPS.
            samples (int): the amount of samples used to estimate the expected value of the twCRPS.

        Returns:
            the twCRPS values for the given thresholds.
        """
        if self.models is None:
            self.load_models()

        twcrps = np.zeros(shape=(len(thresholds), self.num_models))
        for i, model in enumerate(self.models):
            twcrps[:, i] = model.twCRPS(data, thresholds, samples)

        return twcrps
    
    def CRPS(self, data: tf.data.Dataset, samples: int) -> float:
        """
        Compute the CRPS for the given data.

        Arguments:
            data (tf.data.Dataset): the dataset containing the input data and observations.
            samples (int): the amount of samples used to estimate the expected value of the CRPS.

        Returns:
            the CRPS value.
        """
        return self.twCRPS(data, np.array([0.0]), samples)[0]







