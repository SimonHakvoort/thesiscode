import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Concatenate, Conv2D, Flatten
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
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

        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)#, run_eagerly=True)

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
    
    def CRPS(self, dataset, sample_size):
        y_true = []
        y_pred = []

        for X, y in dataset:
            y_true.append(y)
            y_pred.append(self.predict(X))
            
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

    
    def _compute_twCRPS(self, y_true, y_pred, sample_size, chain_function):
        distribution = self.get_distribution(y_pred)

        X_1 = distribution.sample(sample_size)
        X_2 = distribution.sample(sample_size)

        vX_1 = chain_function(X_1)
        vX_2 = chain_function(X_2)

        E_1 = tf.reduce_mean(tf.abs(vX_1 - chain_function(tf.squeeze(y_true))), axis=0)
        E_2 = tf.reduce_mean(tf.abs(vX_2 - vX_1), axis=0)

        return tf.reduce_mean(E_1) - 0.5 * tf.reduce_mean(E_2)
    
    def _loss_twCRPS_sample(self, y_true, y_pred):
        return self._compute_twCRPS(y_true, y_pred, self.sample_size, self.chain_function)
    
    # def twCRPS(self, X, y, sample_size, t):
    #     y_pred = self.predict(X)
    #     return self._compute_twCRPS(y, y_pred, sample_size, lambda x: self._chain_function_indicator(x, t))
    
    # def twCRPS(self, dataset, sample_size, t):
    #     y_true = []
    #     y_pred = []

    #     for X, y in dataset:
    #         y_true.append(y)
    #         y_pred.append(self.predict(X))
            
    #     y_true = tf.concat(y_true, axis=0)
    #     y_pred = tf.concat(y_pred, axis=0)
    #     return self._compute_twCRPS(y_true, y_pred, sample_size, lambda x: self._chain_function_indicator(x, t))

    def twCRPS(self, dataset, thresholds, sample_size):
        y_true = []
        y_pred = []

        for X, y in dataset:
            y_true.append(y)
            y_pred.append(self.predict(X))

        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)

        scores = []
        for threshold in thresholds:
            scores.append(self._compute_twCRPS(y_true, y_pred, sample_size, lambda x: self._chain_function_indicator(x, threshold)))
        return scores
    

    
    # def Brier_Score(self, X, y, threshold):
    #     y_pred = self.predict(X)
    #     distribution = self.get_distribution(y_pred)
    #     cdf_values = distribution.cdf(threshold)
    #     return tf.reduce_mean(tf.square(self.indicator_function(y, threshold) - cdf_values))
    

    ### Checken of de distribution van de GEV distribution niet NaN geeft!!!!!!
    # def Brier_Score(self, dataset, threshold):
    #     y_true = []
    #     y_pred = []

    #     for X, y in dataset:
    #         y_true.append(y)
    #         y_pred.append(self.predict(X))
            
    #     y_true = tf.concat(y_true, axis=0)
    #     y_pred = tf.concat(y_pred, axis=0)
    #     distribution = self.get_distribution(y_pred)
    #     cdf_values = distribution.cdf(threshold)
    #     return tf.reduce_mean(tf.square(self.indicator_function(y_true, threshold) - cdf_values))

    def Brier_Score(self, dataset, thresholds):
        y_true = []
        y_pred = []

        for X, y in dataset:
            y_true.append(y)
            y_pred.append(self.predict(X))
            
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)
        distributions = self.get_distribution(y_pred)
        
        scores = []
        for threshold in thresholds:
            cdf_values = distributions.cdf(threshold)
            scores.append(tf.reduce_mean(tf.square(self.indicator_function(y_true, threshold) - cdf_values)))
        return scores

    
    def indicator_function(self, y, threshold):
        return tf.cast(y <= threshold, tf.float32)
    
    def my_save(self, filepath):
        os.makedirs(filepath + '/nnmodel', exist_ok=True)
        self.model.my_save(filepath + '/nnmodel')

        setup = {
            'sample_size': self.sample_size,
            'features_names': self.features_names,
            'add_wind_conv': self.add_wind_conv,
            'features_1d_mean': self.features_1d_mean,
            'features_1d_std': self.features_1d_std,
        }

        setup['setup_loss'] = {}

        if hasattr(self, 'loss_function'):
            if self.loss_function == self._loss_CRPS_sample:
                setup['setup_loss']['loss_function'] = 'loss_CRPS_sample'
            elif self.loss_function == self._loss_twCRPS_sample:
                setup['setup_loss']['loss_function'] = 'loss_twCRPS_sample'
                if hasattr(self, 'chain_function'):
                    setup['setup_loss']['chain_function'] = self.chain_function.__name__
                    if hasattr(self, 'chain_function_threshold'):
                        setup['setup_loss']['chain_function_threshold'] = self.chain_function_threshold
                    if hasattr(self, 'chain_function_normal_distribution'):
                        setup['setup_loss']['chain_function_mean'] = self.chain_function_normal_distribution.loc.numpy()
                        setup['setup_loss']['chain_function_std'] = self.chain_function_normal_distribution.scale.numpy()
                    if hasattr(self, 'chain_function_constant'):
                        setup['setup_loss']['chain_function_constant'] = self.chain_function_constant

        setup['setup_optimizer'] = {
            'optimizer': self.optimizer.__class__.__name__.lower(),
            'learning_rate': self.optimizer.learning_rate.numpy(),
        }

        with open(filepath + '/attributes', 'wb') as f:
            pickle.dump(setup, f)

    @classmethod
    def my_load(self, filepath):
        with open(filepath + '/attributes', 'rb') as f:
            attributes = pickle.load(f)

        nnforecast = NNForecast(**attributes)

        nnforecast.model = NNModel.my_load(filepath + '/nnmodel')# , make_conv=nnforecast.add_wind_conv)

        nnforecast._init_optimizer(**attributes['setup_optimizer'])

        nnforecast.model.compile(optimizer=nnforecast.optimizer, loss=nnforecast.loss_function)

        return nnforecast

    @classmethod
    def load(cls, filepath):
        with open(filepath + '/attributes', 'rb') as f:
            attributes = pickle.load(f)

        nnforecast = cls(**attributes)

        # let custom objects contain the correct loss function
        custom_objects = {
            '_loss_CRPS_sample': nnforecast._loss_CRPS_sample,
            '_loss_twCRPS_sample': nnforecast._loss_twCRPS_sample,
        }

        nnforecast.model = NNModel.load(filepath + '/nnmodel', **custom_objects)

        nnforecast.model.compile(optimizer=nnforecast.optimizer, loss=nnforecast.loss_function)

        return nnforecast


    
    def fit(self, dataset, epochs=100):

        history = self.model.fit(dataset, epochs=epochs)

        return history

    def predict(self, X):

        return self.model.predict(X)
    
    def get_prob_distribution(self, data):
        y_pred = []
        for X, y in data:
            y_pred.append(self.predict(X))

        y_pred = tf.concat(y_pred, axis=0)
        return self.model._forecast_distribution.get_distribution(y_pred)


