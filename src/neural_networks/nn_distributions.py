import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint

from src.models.probability_distributions import DistributionMixture

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
    

class NNTruncNormal():
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]
        return tfp.distributions.TruncatedNormal(loc=loc, scale=scale, low=0.0, high=1000.0)
    
    def build_output_layers(self):
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma

    
    def add_forecast(self, outputs, inputs):
        return tf.concat([outputs[:, 0:1] + tf.expand_dims(inputs['wind_speed_forecast'], axis=-1), outputs[:, 1:]], axis=1)
    

class NNLogNormal():
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]
        return tfp.distributions.LogNormal(loc=loc, scale=scale)
    
    def build_output_layers(self):
        mu = Dense(1, activation='linear')
        sigma = Dense(1, activation='softplus')
        return mu, sigma
    
    def add_forecast(self, outputs, inputs):
        adjusted_mean = outputs[:, 0:1] + tf.math.log(tf.expand_dims(inputs['wind_speed_forecast'], axis=-1)) - tf.square(outputs[:, 1:])/2
        return tf.concat([adjusted_mean, outputs[:, 1:]], axis=1)
    

class NNGEV():
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1] + 1e-5
        concentration = y_pred[:, 2]
        return tfp.distributions.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration)
    
    def build_output_layers(self):
        loc = Dense(1, activation='linear')
        scale = Dense(1, activation='softplus')
        shape = Dense(1, activation='linear', kernel_constraint=SymmetricClipConstraint(1.0))
        return loc, scale, shape
    
    def add_forecast(self, outputs, inputs):
        return tf.concat([outputs[:, 0:1] + tf.expand_dims(inputs['wind_speed_forecast'], axis=-1), outputs[:, 1:]], axis=1)

class NNMixture():
    def __init__(self, distribution_1, distribution_2):
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
        