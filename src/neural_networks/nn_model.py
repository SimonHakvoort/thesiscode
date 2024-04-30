import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten
import pickle

import pdb

from neural_networks.nn_distributions import NNDistribution, NNTruncNormal

class NNModel(Model):
    def __init__(self, forecast_distribution, **kwargs):
        super(NNModel, self).__init__()

        self._forecast_distribution = forecast_distribution

        if not kwargs:
            return

        self.hidden_layers = []

        for units in kwargs['hidden_units_list']:
            if 'dense_l1_regularization' in kwargs and kwargs["dense_l1_regularization"] != 0 and 'dense_l2_regularization' in kwargs and kwargs["dense_l2_regularization"] != 0:
                regularizer = tf.keras.regularizers.l1_l2(l1=kwargs["dense_l1_regularization"], l2=kwargs["dense_l2_regularization"])
            elif 'dense_l1_regularization' in kwargs and kwargs["dense_l1_regularization"] != 0:
                regularizer = tf.keras.regularizers.l1(kwargs["dense_l1_regularization"])
            elif 'dense_l2_regularization' in kwargs and kwargs["dense_l2_regularization"] != 0:
                regularizer = tf.keras.regularizers.l2(kwargs["dense_l2_regularization"])
            else:
                regularizer = None
            self.hidden_layers.append(Dense(units, activation='relu', kernel_regularizer=regularizer))

        self.output_layers = forecast_distribution.build_output_layers()

        self.concatenate = Concatenate()

        self.setup = {
            'hidden_units_list': kwargs['hidden_units_list'],
            'dense_l1_regularization': kwargs['dense_l1_regularization'],
            'dense_l2_regularization': kwargs['dense_l2_regularization'],
            'forecast_distribution': forecast_distribution,
        }
        

    def call(self, inputs):

        x = Flatten()(inputs['features_1d'])
        
        for layer in self.hidden_layers:
            x = layer(x) 

        
        outputs = self.concatenate([layer(x) for layer in self.output_layers])

        updated_outputs = self._forecast_distribution.add_forecast(outputs, inputs)

        return updated_outputs
    
    def get_forecast_distribution(self):
        return self._forecast_distribution
    
    def get_config(self):
        config = super(NNModel, self).get_config()
        config.update({
            'forecast_distribution': self._forecast_distribution.get_config()
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        forecast_distribution = config.pop('forecast_distribution')
        # forecast_distribution = NNTruncNormal.from_config(forecast_distributions_config)
        return cls(forecast_distribution, **config)
    
    def my_save(self, filepath):
        configuration_path = filepath + '/configuration'
        with open(configuration_path, 'wb') as f:
            pickle.dump(self.setup, f)

        # save the weights 
        self.save_weights(filepath + '/weights')


    @staticmethod
    def my_load(filepath):
        configuration_path = filepath + '/configuration'
        
        with open(configuration_path, 'rb') as f:
            configuration = pickle.load(f)
        
        # forecast_distribution = configuration['forecast_distribution']
        model = NNModel(**configuration)

        model.load_weights(filepath + '/weights')

        return model


        
                
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__() 
        self.dense = Dense(units, **kwargs)

    def call(self, inputs):
        return self.dense(inputs) + inputs



