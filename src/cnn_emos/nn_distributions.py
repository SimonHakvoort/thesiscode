from typing import Tuple, Union
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint

from abc import ABC, abstractmethod


import pdb

def distribution_name(distribution: str, **kwargs):
    """
    Factory function that returns an instance of a specific forecast distribution model based on the provided distribution name.

    Args:
        distribution (str): The name of the distribution. Case insensitive. Supported values are:
                            - 'trunc_normal', 'truncated_normal', 'distr_trunc_normal', 'distr_tn', 'tn', 'truncnorm' for NNTruncNormal.
                            - 'log_normal', 'lognormal', 'distr_log_normal', 'distr_ln', 'ln', 'lognorm' for NNLogNormal.
                            - 'gev', 'generalized_extreme_value', 'distr_gev', 'distr_generalized_extreme_value' for NNGEV.
                            - 'mixture', 'distr_mixture' for NNMixture. Requires 'distribution_1' and 'distribution_2' in kwargs.

        **kwargs: Additional keyword arguments for configuring the NNMixture.

    Returns:
        An instance of a NNDistribution the corresponding forecast distribution model based on the given distribution name.
    """
    if distribution.lower() in ['trunc_normal', 'truncated_normal', 'distr_trunc_normal', 'distr_tn', 'tn', 'truncnorm']:
        return NNTruncNormal()
    elif distribution.lower() in ['log_normal', 'lognormal', 'distr_log_normal', 'distr_ln', 'ln', 'lognorm']:
        return NNLogNormal()
    elif distribution.lower() in ['gev', 'generalized_extreme_value', 'distr_gev', 'distr_generalized_extreme_value']:
        return NNGEV()
    elif distribution.lower() in ['mixture', 'distr_mixture']:
        if 'distribution_1' not in kwargs:
            raise ValueError("distribution_1 must be provided")
        if 'distribution_2' not in kwargs:
            raise ValueError("distribution_2 must be provided")
        return NNMixture(distribution_name(kwargs['distribution_1']), distribution_name(kwargs['distribution_2']))
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    

class SymmetricClipConstraint(Constraint):
    def __init__(self, clip_value: float):
        self.clip_value = clip_value

    def __call__(self, w):
        return tf.clip_by_value(w, -self.clip_value, self.clip_value)
    
class MinMaxConstraint(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)
        
class NNDistribution(ABC):
    """
    Abstract base class for the parametric distributions of the CNNEMOS.
    It can be used together with an NNConvModel to make predictions. 
    Based on the output of the neural network it will make an instance of tfp.distribution.Distribution
    using the get_distribution method. With buid_output_layers we can return the correct layers for the 
    output of the neural network. This is dependent on the specific distribution.
    """
    @abstractmethod
    def get_distribution(self, y_pred: tf.Tensor) -> tfp.distributions.Distribution:
        """
        Based on the output of the CNN it returns the parametric distribution.
        This is an abstract method that needs to get implemented by child classes.

        Arguments:
            y_pred (tf.Tensor): the output of the CNN.

        Returns:
            The distribution (tfp.distributions.Distribution)
        """
        pass
    
    @abstractmethod
    def build_output_layers(self):
        """
        Builds the output layers of the neural network, which depends on the specific parametric distribution.
        This is an abstract method that needs to get implemented by child classes.

        Returns:
            A tuple of tf.keras.layers.Dense layers.
        """
        pass
    
    @abstractmethod
    def __str__(self):
        """
        Returns the name of the parametric distribution
        """
        pass

    @abstractmethod
    def short_name(self) -> str:
        """
        Returns a shortened version of the name of the distribution, which can be used for saving.

        Returns:
            The shortened name of the distribution.
        """
        pass
    
    def has_gev(self) -> bool:
        """
        Checks whether the parametric distribution contains a GEV distribution.
        """
        return False
    
    def is_mixture(self) -> bool:
        """
        Checks whether this is an mixture distribution.
        """
        return False
    
    def comp_cdf(self, y_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative distribution function of the forecast distribution for a range of values.
        
        Args:
            y_pred (np.ndarray): the predicted parameters of the distribution.
            values (np.ndarray): values for which to compute the cdf.

        Returns:
            np.ndarray: The cdf values for each value in the input array, with shape (len(values), num_samples)
        """
        output = np.zeros((len(values), y_pred.shape[0]))
        distr = self.get_distribution(y_pred)
        for i, value in enumerate(values):
            output[i] = distr.cdf(value)

        return output

    
class NNTruncNormal(NNDistribution):
    """
    Class representing a truncated normal distribution for the CNNEMOS model.
    """
    @staticmethod
    def get_distribution(y_pred: tf.Tensor) -> tfp.distributions.TruncatedNormal:
        """
        Returns the TruncatedNormal distribution based on the CNN output.

        Arguments:
            y_pred (tf.Tensor): The output of the CNN containing the parameters for the TruncatedNormal distribution.

        Returns:
            tfp.distributions.TruncatedNormal: The TruncatedNormal distribution parameterized by the CNN output.
        """
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]

        return tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=1000.0)
    
    def build_output_layers(self) -> Tuple[Dense, Dense]:
        """
        Constructs and returns the output layers for the neural network corresponding to the truncated normal distribution.

        Returns:
            Tuple[Dense, Dense]: The Dense layers for the mean (mu) and standard deviation (sigma) of the TruncatedNormal distribution.
        """
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma
    
    def __str__(self) -> str:
        """
        Returns the name of the distribution.

        Returns:
            str: The name of the distribution.
        """
        return "TruncNormal"
    
    def short_name(self) -> str:
        """
        Returns a short name for the distribution.

        Returns:
            str: The short name of the distribution.
        """
        return "tn"


class NNLogNormal(NNDistribution):
    """
    Class representing a log-normal distribution for the CNNEMOS model.
    """
    def get_distribution(self, y_pred: tf.Tensor) -> tfp.distributions.LogNormal: 
        """
        Returns the LogNormal distribution based on the CNN output.

        Arguments:
            y_pred (tf.Tensor): The output of the CNN containing the parameters for the LogNormal distribution.

        Returns:
            tfp.distributions.LogNormal: The LogNormal distribution parameterized by the CNN output.
        """
        mean = y_pred[:, 0]
        var = y_pred[:, 1]

        # The loc and scale follow this transformation to ensure that the resulting 
        # log-normal distribution has expectation mean and variance var
        loc = tf.math.log(mean ** 2) - 0.5 * tf.math.log(var + mean ** 2)

        # We add a small value to avoid numerical instability
        scale = tf.sqrt(tf.math.log(1 + var / mean ** 2) + 1e-6)

        return tfp.distributions.LogNormal(loc=loc, scale=scale)
    
    def build_output_layers(self) -> Tuple[Dense, Dense]:
        """
        Constructs and returns the output layers for the neural network corresponding to the log-normal distribution.

        Returns:
            Tuple[Dense, Dense]: The Dense layers for the mean (mu) and standard deviation (sigma) of the LogNormal distribution.
        """
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma
    

    def __str__(self) -> str:
        """
        Returns the name of the distribution.

        Returns:
            str: The name of the distribution.
        """
        return "LogNormal"
    

    def short_name(self) -> str:
        """
        Returns a short name for the distribution.

        Returns:
            str: The short name of the distribution.
        """
        return "ln"
    

class NNGEV(NNDistribution):
    """
    Class representing a Generalized Extreme Value (GEV) distribution for the CNNEMOS model.
    Has additional methods since tfp.distributions.GeneralizedExtremeValue returns NaN for the methods prob or cdf outside of its domain.
    """
    def get_distribution(self, y_pred: tf.Tensor) -> tfp.distributions.GeneralizedExtremeValue:
        """
        Returns the GEV distribution based on the CNN output.

        Arguments:
            y_pred (tf.Tensor): The output of the CNN containing the parameters for the GEV distribution.

        Returns:
            tfp.distributions.GeneralizedExtremeValue: The GEV distribution parameterized by the CNN output.
        """
        loc = y_pred[:, 0]
        scale = y_pred[:, 1] + 1e-5
        concentration = y_pred[:, 2]
        return tfp.distributions.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration)
    
    def build_output_layers(self) -> Tuple[Dense, Dense, Dense]:
        """
        Constructs and returns the output layers for the neural network corresponding to the GEV distribution.

        Returns:
            Tuple[Dense, Dense, Dense]: The Dense layers for the location (loc), scale, and concentration parameters of the GEV distribution.
        """
        # Setting these constraint results in better convergence.
        loc = Dense(1, activation='linear', kernel_constraint=MinMaxConstraint(-10, 30))
        scale = Dense(1, activation='softplus', kernel_constraint=SymmetricClipConstraint(5))
        shape = Dense(1, activation='linear', kernel_constraint=SymmetricClipConstraint(0.5))
        return loc, scale, shape
    
    def __str__(self) -> str:
        """
        Returns the name of the distribution.

        Returns:
            str: The name of the distribution.
        """
        return "GEV"
    
    def short_name(self) -> str:
        """
        Returns a short name for the distribution.

        Returns:
            str: The short name of the distribution.
        """
        return "gev"
    
    def has_gev(self) -> bool:
        """
        Checks whether the distribution contains a GEV distribution, which is True in this case.

        Returns:
            bool: True, since this is a GEV distribution.
        """
        return True
    
    def get_gev_shape(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Returns the shape parameter of the GEV distribution based on the predicted values.

        Arguments:
            y_pred (tf.Tensor): The predicted values.

        Returns:
            tf.Tensor: The shape parameter of the GEV distribution.
        """
        return y_pred[:, 2]
    
    def comp_cdf(self, y_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Computes the CDF values for the distribution parameters y_pred at values.
        Based on the CDF values, we need to change the NaNs into 0 or 1 depending on the shape of the distribution.

        Arguments:
            y_pred (np.ndarray): tensor of the predicted values.
            values (np.ndarray): the values at which we want to find the CDF.

        Returns:
            The correct cdf at  values.
        """
        shape = self.get_gev_shape(y_pred)

        output = np.zeros((len(values), y_pred.shape[0]))
        distr = self.get_distribution(y_pred)

        for i, value in enumerate(values):
            cdf_value = distr.cdf(value).numpy()
            nan_indices = np.isnan(cdf_value)  # Find NaN indices
            shape_less_than_zero = shape < 0  # Find shape < 0 indices
            output[i] = cdf_value

            # Update NaN values based on the condition
            output[i, nan_indices & shape_less_than_zero] = 1
            output[i, nan_indices & ~shape_less_than_zero] = 0

        return output

class NNMixture(NNDistribution):
    """
    Class for the Mixture distribution for CNNEMOS. It contains two underlying NNDistributions and combines these two with an extra weight parameter.
    """
    def __init__(self, distribution_1: NNDistribution, distribution_2: NNDistribution):
        if not isinstance(distribution_1, NNDistribution):
            raise ValueError("distribution_1 must be an instance of NNDistribution")
        
        if not isinstance(distribution_2, NNDistribution):
            raise ValueError("distribution_2 must be an instance of NNDistribution")

        self.distribution_1 = distribution_1
        self.distribution_2 = distribution_2
        self.num_params_distribution_1 = len(distribution_1.build_output_layers())
        self.num_params_distribution_2 = len(distribution_2.build_output_layers())
    
    def get_distribution(self, y_pred: tf.Tensor) -> tfp.distributions.Mixture:
        """
        Based on y_pred it returns the mixture distribution.

        Arguments:
            y_pred (tf.Tensor): the predicted parameters of the distribution.

        Returns:
            tfp.distributions.Mixture with the parameters from y_pred.
        """
        # The first parameter is the weight, then the parameters from distribution_1 and the final parameters are from distribution_2.
        weight = y_pred[:, 0]
        params_1 = y_pred[:, 1:1+self.num_params_distribution_1]
        params_2 = y_pred[:, 1+self.num_params_distribution_1:]
        
        dist_1 = self.distribution_1.get_distribution(params_1)
        dist_2 = self.distribution_2.get_distribution(params_2)

        cat = tfp.distributions.Categorical(probs=tf.stack([weight, 1 - weight], axis=-1))

        return tfp.distributions.Mixture(
            cat,
            [dist_1, dist_2]
        )
    
    def build_output_layers(self):
        """
        Returns the output layers for this specific distribuion. The first one is the weight parameter, which has sigmoid activation.
        Then the layers from distribution_1 and distribution_2 follow.

        Returns:
            The output layers for the NNMixture distribution.
        """
        weight = Dense(1, activation='sigmoid')
        params_1 = self.distribution_1.build_output_layers()
        params_2 = self.distribution_2.build_output_layers()
        return weight, *params_1, *params_2

    def __str__(self):
        return f"Mixture({self.distribution_1}, {self.distribution_2})"
    
    def short_name(self):
        return f"mix_{self.distribution_1.short_name()}_{self.distribution_2.short_name()}"
    
    def has_gev(self) -> bool:
        """
        Checks whether the underlying distributions are the GEV distribution.
        """
        return self.distribution_1.has_gev() or self.distribution_2.has_gev()
    
    def get_gev_shape(self, y_pred: tf.Tensor) -> Union[tf.Tensor, None]:
        """
        Returns the shape of the GEV distribution in case this is part of distribution_1 or distribution_2.
    
        Arguments:
            y_pred (tf.Tensor): the predicted parameters of the distribution.

        Returns:
            The shape of the GEV distribution (tf.Tensor) in case this is applicable, otherwise None.
        """
        if self.distribution_1.has_gev():
            return self.distribution_1.get_gev_shape(y_pred[:, 1:1+self.num_params_distribution_1])
        elif self.distribution_2.has_gev():
            return self.distribution_2.get_gev_shape(y_pred[:, 1+self.num_params_distribution_1:])
        else:
            return None
        
    def is_mixture(self) -> bool:
        """
        Checks whether this is an Mixture distribution.
        """
        return True
    
    def get_weight(self, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Returns the weight of the mixture distribution for the predicted parameters.

        Arguments:
            y_pred (tf.Tensor): the predicted parameters.

        Returns:
            The weight of the distributions (tf.Tensor).
        """
        return y_pred[:, 0]
    
    def get_shape_and_weight(self, y_pred: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Returns the shape and the weight of the predicted parameters. Only applicable in case has_gev is True.

        Arguments:
            y_pred (tf.Tensor): the predicted parameters of the distribution.

        Returns:
            A tuple of the shape of the GEV and the weight.
        """
        if not self.has_gev():
            return None
        
        gev_shape = None
        if self.distribution_1.has_gev():
            gev_shape = self.distribution_1.get_gev_shape(y_pred[:, 1:1+self.num_params_distribution_1])
        elif self.distribution_2.has_gev():
            gev_shape = self.distribution_2.get_gev_shape(y_pred[:, 1+self.num_params_distribution_1:])

        weight = y_pred[:, 0]

        return gev_shape, weight
    
    def comp_cdf(self, y_pred: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Computes the CDF values for the distribution parameters y_pred at values.

        Arguments:
            y_pred (np.ndarray): tensor of the predicted values.
            values (np.ndarray): the values at which we want to find the CDF.

        Returns:
            The correct cdf at  values.
        """
        weight = self.get_weight(y_pred)

        values_1 = self.distribution_1.comp_cdf(y_pred[:, 1:1+self.num_params_distribution_1], values)
        values_2 = self.distribution_2.comp_cdf(y_pred[:, 1+self.num_params_distribution_1:], values)

        return weight * values_1 + (1 - weight) * values_2