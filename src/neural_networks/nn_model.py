import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle

import pdb

from src.neural_networks.nn_distributions import NNDistribution

class NNBaseModel(Model):
    
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



class NNConvModel(NNBaseModel):
    """
    Convolutional Neural Network (CNN) model implemented with Keras.

    The attribute _forecast_distribution contains the specific forecast distribution of the model.

    It uses 7x7, 5x5 and 3x3 convolutional blocks, each followed by batch normalization.
    The output of the final convolutional block is concatenated with 'features_1d'. Then there are 
    dense blocks, followed by an output which is dependent in the parametric distribution of the forecast.
    """
    def __init__(self, forecast_distribution, **kwargs):
        """
        Initialize the CNN model with given forecast distribution and optional keyword arguments.

        kwargs should contain the following keys:
            hidden_units_list (list): List of units for hidden dense layers. Each element stands for a single hidden layer.
            dense_l1_regularization (float): L1 regularization factor for dense layers.
            dense_l2_regularization (float): L2 regularization factor for dense layers.
            conv_7x7_units (int): Number of 7x7 convolutional units.
            conv_5x5_units (int): Number of 5x5 convolutional units.
            conv_3x3_units (int): Number of 3x3 convolutional units.
        """
        super(NNConvModel, self).__init__()

        self._forecast_distribution = forecast_distribution

        if not kwargs:
            return

        # A list containing the dense layers.
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

        # Used to concatenate the output of the convolutional block with 'features_1d'.
        self.concatenate = Concatenate()

        # Initialize the 7x7, 5x5 and 3x3 convolutions. After every convolution we apply batch normalization
        # After each convolutional block we apply 2x2 max pooling.
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
            'dense_l2_regularization': kwargs['dense_l2_regularization'],
            'forecast_distribution': forecast_distribution,
            'conv_7x7_units': kwargs['conv_7x7_units'],
            'conv_5x5_units': kwargs['conv_5x5_units'],
            'conv_3x3_units': kwargs['conv_3x3_units'],
        }


    def call(self, inputs: dict[str: tf.Tensor]) -> tf.Tensor:
        """
        Perform the forward pass of the CNN model.

        Args:
            inputs (dict): Dictionary containing input tensors:
                           - 'wind_speed_grid': Grid input for wind speed.
                           - 'features_1d': 1D features input.
                           - 'wind_speed_forecast': Wind speed forecast input.

        Returns:
            tf.Tensor: Output tensor from the model.
        """
        # Extract the features from the data.
        grid_input = inputs['wind_speed_grid']
        features_1d = inputs['features_1d']
        wind_speed_forecast = inputs['wind_speed_forecast']

        x = grid_input

        # # Put the grid_input through the convolutional block.
        # for layer, batch_norm in zip(self.conv_7x7_layers, self.batch_norm_7x7_layers):
        #     x = layer(x)
        #     x = batch_norm(x)

        # x = self.max_pooling_2x2(x)

        for layer, batch_norm in zip(self.conv_5x5_layers, self.batch_norm_5x5_layers):
            x = layer(x)
            x = batch_norm(x)

        x = self.max_pooling_2x2(x)

        for layer, batch_norm in zip(self.conv_3x3_layers, self.batch_norm_3x3_layers):
            x = layer(x)
            x = batch_norm(x)

        x = self.max_pooling_2x2(x)

        x = Flatten()(x)

        # Concatenate the inputs and put it through the dense layers.
        x = Concatenate()([x, Flatten()(features_1d)])

        for layer in self.hidden_layers:
            x = layer(x)

        # Concatenate the output of the final layer with the wind_speed_forecast.
        x = Concatenate()([x, Flatten()(wind_speed_forecast)])

        # Each output is an element of output_layers. This is dependend on the parametric distribution.
        outputs = self.concatenate([layer(x) for layer in self.output_layers])
        
        return outputs
    
    def has_gev(self) -> bool:
        """
        Check if the forecast distribution contains the Generalized Extreme Value (GEV) distribution.

        Returns:
            bool: True if the forecast distribution contains GEV, False otherwise.
        """
        return self._forecast_distribution.has_gev()
    
    def get_forecast_distribution(self) -> NNDistribution:
        """
        Returns the parametric distribution (NNDistribution).

        Returns:
            self._forecast_distribution (NNDistribution).
        """
        return self._forecast_distribution
    