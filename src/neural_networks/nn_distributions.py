import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint

def distribution_name(distribution):
    if distribution.lower() in ['trunc_normal', 'truncated_normal', 'distr_trunc_normal', 'distr_tn', 'tn', 'truncnorm']:
        return NNTruncNormal()
    elif distribution.lower() in ['log_normal', 'lognormal', 'distr_log_normal', 'distr_ln', 'ln', 'lognorm']:
        return NNLogNormal()
    elif distribution.lower() in ['gev', 'generalized_extreme_value', 'distr_gev', 'distr_generalized_extreme_value']:
        return NNGEV()
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
    

class NNLogNormal():
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1]
        return tfp.distributions.LogNormal(loc=loc, scale=scale)
    
    def build_output_layers(self, final_layer):
        mu = Dense(1, activation='linear')(final_layer)
        sigma = Dense(1, activation='softplus')(final_layer)
        return mu, sigma
    

class NNGEV():
    def get_distribution(self, y_pred):
        loc = y_pred[:, 0]
        scale = y_pred[:, 1] + 1e-5
        concentration = y_pred[:, 2]
        return tfp.distributions.GeneralizedExtremeValue(loc=loc, scale=scale, concentration=concentration)
    
    def build_output_layers(self, final_layer):
        loc = Dense(1, activation='linear')(final_layer)
        scale = Dense(1, activation='softplus')(final_layer)
        shape = Dense(1, activation='linear', kernel_constraint=SymmetricClipConstraint(1.0))(final_layer)
        return loc, scale, shape