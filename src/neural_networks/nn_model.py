import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten

class NNModel(Model):
    def __init__(self, forecast_distribution, hidden_units_list, input_shape, dense_l2_regularization):
        super(NNModel, self).__init__()

        self.hidden_layers = []

        for units in hidden_units_list:
            self.hidden_layers.append(Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(dense_l2_regularization)))

        self.output_layers = forecast_distribution.build_output_layers()

        self.concatenate = Concatenate()
    
    def call(self, inputs):
        x = Flatten()(inputs)

        for layer in self.hidden_layers:
            x = layer(x)

        outputs = self.concatenate([layer(x) for layer in self.output_layers])

        return outputs

        
    



