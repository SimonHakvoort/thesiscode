import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle

import pdb

from neural_networks.nn_distributions import NNDistribution, NNTruncNormal

class NNBaseModel(Model):
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
    
    def get_forecast_distribution(self):
        return self._forecast_distribution


class NNModel(NNBaseModel):
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

        self.add_nwp_forecast = kwargs['add_nwp_forecast']

        self.setup = {
            'hidden_units_list': kwargs['hidden_units_list'],
            'dense_l1_regularization': kwargs['dense_l1_regularization'],
            'dense_l2_regularization': kwargs['dense_l2_regularization'],
            'forecast_distribution': forecast_distribution,
            'add_nwp_forecast': kwargs['add_nwp_forecast']
        }

        
        

    def call(self, inputs):

        x = Flatten()(inputs['features_1d'])
        
        for layer in self.hidden_layers:
            x = layer(x) 

        
        outputs = self.concatenate([layer(x) for layer in self.output_layers])

        if self.add_nwp_forecast:
            return self._forecast_distribution.add_forecast(outputs, inputs)

        return outputs

    
    # def get_config(self):
    #     config = super(NNModel, self).get_config()
    #     config.update({
    #         'forecast_distribution': self._forecast_distribution.get_config()
    #     })
    #     return config
    
    # @classmethod
    # def from_config(cls, config):
    #     forecast_distribution = config.pop('forecast_distribution')
    #     # forecast_distribution = NNTruncNormal.from_config(forecast_distributions_config)
    #     return cls(forecast_distribution, **config)



class NNConvModel(NNBaseModel):
    def __init__(self, forecast_distribution, **kwargs):
        super(NNConvModel, self).__init__()

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

        self.add_nwp_forecast = kwargs['add_nwp_forecast']

        self.conv_7x7_layers = []
        self.conv_5x5_layers = []
        self.conv_3x3_layers = []

        self.batch_norm_7x7_layers = []
        self.batch_norm_5x5_layers = []
        self.batch_norm_3x3_layers = []


        for _ in range(kwargs['conv_7x7_units']):
            self.conv_7x7_layers.append(Conv2D(8, (7, 7), activation='relu', padding='same'))
            self.batch_norm_7x7_layers.append(BatchNormalization())

        for _ in range(kwargs['conv_5x5_units']):
            self.conv_5x5_layers.append(Conv2D(16, (5, 5), activation='relu', padding='same'))
            self.batch_norm_5x5_layers.append(BatchNormalization())

        for _ in range(kwargs['conv_3x3_units']):
            self.conv_3x3_layers.append(Conv2D(32, (3, 3), activation='relu', padding='same'))
            self.batch_norm_3x3_layers.append(BatchNormalization())

        self.max_pooling_2x2 = MaxPooling2D((2, 2))

        

        self.setup = {
            'hidden_units_list': kwargs['hidden_units_list'],
            'dense_l1_regularization': kwargs['dense_l1_regularization'],
            'dense_l2_regularization': kwargs['dense_l2_regularization'],
            'forecast_distribution': forecast_distribution,
            'add_nwp_forecast': kwargs['add_nwp_forecast'],
            'conv_7x7_units': kwargs['conv_7x7_units'],
            'conv_5x5_units': kwargs['conv_5x5_units'],
            'conv_3x3_units': kwargs['conv_3x3_units'],
        }


    def call(self, inputs):
        grid_input = inputs['wind_speed_grid']
        features_1d = inputs['features_1d']

        x = grid_input

        for layer, batch_norm in zip(self.conv_7x7_layers, self.batch_norm_7x7_layers):
            x = layer(x)
            x = batch_norm(x)

        x = self.max_pooling_2x2(x)

        for layer, batch_norm in zip(self.conv_5x5_layers, self.batch_norm_5x5_layers):
            x = layer(x)
            x = batch_norm(x)

        x = self.max_pooling_2x2(x)

        for layer, batch_norm in zip(self.conv_3x3_layers, self.batch_norm_3x3_layers):
            x = layer(x)
            x = batch_norm(x)

        x = self.max_pooling_2x2(x)

        x = Flatten()(x)

        x = Concatenate()([x, Flatten()(features_1d)])

        for layer in self.hidden_layers:
            x = layer(x)

        outputs = self.concatenate([layer(x) for layer in self.output_layers])

        if self.add_nwp_forecast:
            return self._forecast_distribution.add_forecast(outputs, inputs)
        
        return outputs
    


      
                
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__() 
        self.dense = Dense(units, **kwargs)

    def call(self, inputs):
        return self.dense(inputs) + inputs



