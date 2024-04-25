import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten

import pdb

class NNModel(Model):
    def __init__(self, forecast_distribution, **kwargs):
        super(NNModel, self).__init__()

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

        self.forecast_distribution = forecast_distribution

    
    def call(self, inputs):

        x = Flatten()(inputs['features_1d'])
        
        for layer in self.hidden_layers:
            x = layer(x) 

        
        outputs = self.concatenate([layer(x) for layer in self.output_layers])

        # we add inputs['wind_speed_forecast'] to the first element of outputs
        #updated_outputs = tf.concat([outputs[:,0] + inputs['wind_speed_forecast'], outputs[:,1]], axis=1)
        # updated_outputs = tf.concat([outputs[:, 0:1] + tf.expand_dims(inputs['wind_speed_forecast'], axis=-1), outputs[:, 1:]], axis=1)
        
        updated_outputs = self.forecast_distribution.add_forecast(outputs, inputs)

        #pdb.set_trace()
        return updated_outputs

        
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__() 
        self.dense = Dense(units, **kwargs)

    def call(self, inputs):
        return self.dense(inputs) + inputs



