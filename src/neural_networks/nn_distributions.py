import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.utils import register_keras_serializable
import pickle

from abc import ABC, abstractmethod


import pdb

def distribution_name(distribution, **kwargs):
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
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, w):
        return tf.clip_by_value(w, -self.clip_value, self.clip_value)
    
class MinMaxConstraint(Constraint):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)
        
@register_keras_serializable(package='Custom')
class NNDistribution(ABC):
    @abstractmethod
    def get_distribution(self, y_pred):
        pass
    
    @abstractmethod
    def build_output_layers(self):
        pass
    
    @abstractmethod
    def add_forecast(self, outputs, inputs):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def short_name(self):
        pass
    
    @classmethod
    def from_config(cls, config):
        return NNTruncNormal.from_config(config)
        
    def get_config(self):
        return {'name': self.__str__()}
    
    def has_gev(self):
        return False
    
    def is_mixture(self):
        return False
    
    def comp_cdf(self, y_pred: tf.Tensor, values: np.ndarray) -> np.ndarray:
        """
        Computes the cumulative distribution function of the forecast distribution for a range of values.
        
        Args:
        - X (tf.Tensor): Input data
        - values (np.ndarray): Values for which to compute the cdf

        Returns:
        - np.ndarray: The cdf values for each value in the input array, with shape (len(values), num_samples)
        """
        output = np.zeros((len(values), y_pred.shape[0]))
        distr = self.get_distribution(y_pred)
        for i, value in enumerate(values):
            output[i] = distr.cdf(value).numpy()

        return output

    
@register_keras_serializable(package='Custom')
class NNTruncNormal(NNDistribution):
    @staticmethod
    def get_distribution(y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]
        return tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=1000.0)
    
    def build_output_layers(self):
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma

    def add_forecast(self, outputs, inputs):
        return tf.concat([outputs[:, 0:1] + tf.expand_dims(inputs['wind_speed_forecast'], axis=-1), outputs[:, 1:]], axis=1)
    
    def __str__(self):
        return "TruncNormal"
    
    def short_name(self):
        return "tn"
    
    def get_config(self):
        return {}
    
    @classmethod
    def from_config(cls, config):
        return cls()
    
    

class NNLogNormal(NNDistribution):
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]

        mean = tf.math.log(loc ** 2) - 0.5 * tf.math.log(scale + loc ** 2)

        # We add a small value to avoid numerical instability
        sigma = tf.sqrt(tf.math.log(1 + scale / loc ** 2) + 1e-6)

        return tfp.distributions.LogNormal(loc=mean, scale=sigma)
    
    def build_output_layers(self):
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma
    
    def add_forecast(self, outputs, inputs):
        adjusted_mean = outputs[:, 0:1] + tf.math.log(tf.expand_dims(inputs['wind_speed_forecast'], axis=-1)) - tf.square(outputs[:, 1:])/2
        return tf.concat([adjusted_mean, outputs[:, 1:]], axis=1)
    
    def __str__(self):
        return "LogNormal"
    
    def short_name(self):
        return "ln"
    

class NNGEV(NNDistribution):
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1] + 1e-5
        concentration = y_pred[:, 2]
        return tfp.distributions.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration)
    
    def build_output_layers(self):
        # Setting these constraint results in better convergence.
        loc = Dense(1, activation='linear', kernel_constraint=MinMaxConstraint(-10, 30))
        scale = Dense(1, activation='softplus', kernel_constraint=SymmetricClipConstraint(5))
        shape = Dense(1, activation='linear', kernel_constraint=SymmetricClipConstraint(0.5))
        return loc, scale, shape
    
    def add_forecast(self, outputs, inputs):
        return tf.concat([outputs[:, 0:1] + tf.expand_dims(inputs['wind_speed_forecast'], axis=-1), outputs[:, 1:]], axis=1)
    
    def __str__(self):
        return "GEV"
    
    def short_name(self):
        return "gev"
    
    def has_gev(self):
        return True
    
    def get_gev_shape(self, y_pred):
        return y_pred[:, 2]
    
    def comp_cdf(self, y_pred: tf.Tensor, values: np.ndarray) -> np.ndarray:
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
    def __init__(self, distribution_1, distribution_2):
        if not isinstance(distribution_1, NNDistribution):
            raise ValueError("distribution_1 must be an instance of NNDistribution")
        
        if not isinstance(distribution_2, NNDistribution):
            raise ValueError("distribution_2 must be an instance of NNDistribution")

        self.distribution_1 = distribution_1
        self.distribution_2 = distribution_2
        self.num_params_distribution_1 = len(distribution_1.build_output_layers())
        self.num_params_distribution_2 = len(distribution_2.build_output_layers())
    
    def get_distribution(self, y_pred):
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
        weight = Dense(1, activation='sigmoid')
        params_1 = self.distribution_1.build_output_layers()
        params_2 = self.distribution_2.build_output_layers()
        return weight, *params_1, *params_2
    
    def add_forecast(self, outputs, inputs):
        return tf.concat([outputs[:, 0:1], self.distribution_1.add_forecast(outputs[:, 1:1+self.num_params_distribution_1], inputs), self.distribution_2.add_forecast(outputs[:, 1+self.num_params_distribution_1:], inputs)], axis=1)
        

    def __str__(self):
        return f"Mixture({self.distribution_1}, {self.distribution_2})"
    
    def short_name(self):
        return f"mix_{self.distribution_1.short_name()}_{self.distribution_2.short_name()}"
    
    def has_gev(self):
        return self.distribution_1.has_gev() or self.distribution_2.has_gev()
    
    def get_gev_shape(self, y_pred):
        if self.distribution_1.has_gev():
            return self.distribution_1.get_gev_shape(y_pred[:, 1:1+self.num_params_distribution_1])
        elif self.distribution_2.has_gev():
            return self.distribution_2.get_gev_shape(y_pred[:, 1+self.num_params_distribution_1:])
        
    def is_mixture(self):
        return True
    
    def get_weight(self, y_pred):
        return y_pred[:, 0]
    
    def get_shape_and_weight(self, y_pred):
        if not self.has_gev():
            return False
        
        gev_shape = None
        if self.distribution_1.has_gev():
            gev_shape = self.distribution_1.get_gev_shape(y_pred[:, 1:1+self.num_params_distribution_1])
        elif self.distribution_2.has_gev():
            gev_shape = self.distribution_2.get_gev_shape(y_pred[:, 1+self.num_params_distribution_1:])

        weight = y_pred[:, 0]

        return gev_shape, weight
    
    def comp_cdf(self, y_pred: tf.Tensor, values: np.ndarray) -> np.ndarray:
        weight = self.get_weight(y_pred).numpy()

        values_1 = self.distribution_1.comp_cdf(y_pred, values)
        values_2 = self.distribution_2.comp_cdf(y_pred, values)

        return weight * values_1 + (1 - weight) * values_2