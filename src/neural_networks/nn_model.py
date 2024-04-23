import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten

import pdb

class NNModel(Model):
    def __init__(self, forecast_distribution, **kwargs):
        super(NNModel, self).__init__()

        self.hidden_layers = []

        for units in kwargs['hidden_units_list']:
            self.hidden_layers.append(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(kwargs["dense_l2_regularization"])))

        self.output_layers = forecast_distribution.build_output_layers()

        self.concatenate = Concatenate()

        self.add_forecast_layer = False
        if 'add_forecast_layer' in kwargs:
            self.add_forecast_layer = kwargs['add_forecast_layer']
            self.forecast_distribution = forecast_distribution

    
    def call(self, inputs):

        x = Flatten()(inputs)

        inputs = x
        
        for layer in self.hidden_layers:
            x = layer(x) 

        
        outputs = self.concatenate([layer(x) for layer in self.output_layers])
        # pdb.set_trace()
        # outputs = self.forecast_distribution.add_forecast_layers(outputs, inputs)

        # pdb.set_trace()

        # outputs = tf.stack(outputs, axis=1)

        return outputs

        
class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__() 
        self.dense = Dense(units, **kwargs)

    def call(self, inputs):
        return self.dense(inputs) + inputs



