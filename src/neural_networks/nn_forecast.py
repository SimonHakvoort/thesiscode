import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

from src.neural_networks.nn_distributions import distribution_name
from src.neural_networks.nn_model import NNModel, NNModel

import pdb

class NNForecast:
    def __init__(self, **kwargs):
        self._init_distribution(**kwargs['setup_distribution'])

        self.sample_size = kwargs['sample_size']
        self.features_names = kwargs['feature_names']
        self.scaler = StandardScaler() 

        self._init_loss_function(**kwargs['setup_loss'])

        self.model = NNModel(self.forecast_distribution, **kwargs['setup_nn_architecture'])

        self._init_optimizer(**kwargs['setup_optimizer'])

        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)#, run_eagerly=True)

    def _init_distribution(self, **kwargs):
        if 'forecast_distribution' not in kwargs:
            raise ValueError("forecast_distribution must be provided")
        else:
            self.forecast_distribution = distribution_name(kwargs['forecast_distribution'], **kwargs)


    def _init_loss_function(self, **kwargs):
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

        

    def _compute_CRPS(self, y_true, y_pred, sample_size):
        distribution = self.forecast_distribution.get_distribution(y_pred)

        sample_shape = tf.concat([[sample_size], tf.ones_like(distribution.batch_shape_tensor(), dtype=tf.int32)], axis=0)

        X_1 = distribution.sample(sample_shape)
        X_2 = distribution.sample(sample_shape)

        E_1 = tf.reduce_mean(tf.abs(X_1 - tf.squeeze(y_true)), axis=0)
        E_2 = tf.reduce_mean(tf.abs(X_1 - X_2), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
        

    def _loss_CRPS_sample(self, y_true, y_pred):
        return self._compute_CRPS(y_true, y_pred, self.sample_size)

    
    def CRPS(self, X, y, sample_size):
        y_pred = self.predict(X)
        return self._compute_CRPS(y, y_pred, sample_size)
    
    def _chain_function_indicator(self, x, threshold):
        return tf.maximum(x, threshold)

    
    def _chain_function_indicator_for_twCRPS(self, x):
        return self.chain_function_indicator(x, self.chain_function_threshold)
    
        
    def _chain_function_normal_cdf(self, x, normal_distribution):
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part 
    
    def _chain_function_normal_cdf_for_twCRPS(self, x):
        return self.chain_function_normal_cdf(x, self.chain_function_normal_distribution)

    def _chain_function_normal_cdf_plus_constant(self, x, normal_distribution, constant):
        first_part = (x - normal_distribution.loc) * normal_distribution.cdf(x)
        second_part = normal_distribution.scale ** 2 * normal_distribution.prob(x)
        return first_part + second_part + x * constant
    
    def _chain_function_normal_cdf_plus_constant_for_twCRPS(self, x):
        return self._chain_function_normal_cdf_plus_constant(x, self.chain_function_normal_distribution, self.chain_function_constant)

    
    def _compute_twCRPS(self, y_true, y_pred, sample_size, chain_function):
        distribution = self.forecast_distribution.get_distribution(y_pred)

        X_1 = distribution.sample(sample_size)
        X_2 = distribution.sample(sample_size)

        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)

        E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(tf.squeeze(y_true))), axis=0)
        E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def _loss_twCRPS_sample(self, y_true, y_pred):
        return self._compute_twCRPS(y_true, y_pred, self.sample_size, self.chain_function)
    
    def twCRPS(self, X, y, sample_size, t):
        y_pred = self.predict(X)
        return self._compute_twCRPS(y, y_pred, sample_size, lambda x: self._chain_function_indicator(x, t))
    
    def Brier_Score(self, X, y, threshold):
        y_pred = self.predict(X)
        distribution = self.forecast_distribution.get_distribution(y_pred)
        cdf_values = distribution.cdf(threshold)
        return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - cdf_values))
    
    def indicator_function(self, y, threshold):
        return tf.cast(y <= threshold, tf.float32)
    
    def fit(self, X, y, epochs=100, batch_size=32):
        X = self.scaler.fit_transform(X)

        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

        return history

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.model.predict(X)
